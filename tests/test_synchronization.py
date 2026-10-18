"""Camera time-offset estimation (audio GCC-PHAT and visual events)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from ml.calibration.camera import look_at_camera
from ml.synchronization import SyncResult, apply_sync, audio_offset, audio_offset_from_signals, event_offset
from ml.video.ingest import write_video

SR = 16000


def _clicks(duration_s: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(int(duration_s * SR))
    for t in rng.uniform(0.1, duration_s - 0.1, 30):
        i = int(t * SR)
        x[i: i + 160] += np.hanning(160) * np.sin(np.arange(160) * 1.3) * rng.uniform(0.3, 0.9)
    return x


def _write_wav(path: Path, x: np.ndarray, rate: int = SR) -> Path:
    c = av.open(str(path), "w")
    s = c.add_stream("pcm_s16le", rate=rate, layout="mono")
    pcm = (np.clip(x, -1, 1) * 32000).astype(np.int16)
    for k in range(0, len(pcm), 1024):
        fr = av.AudioFrame.from_ndarray(pcm[None, k: k + 1024], format="s16", layout="mono")
        fr.sample_rate = rate
        fr.pts, fr.time_base = k, Fraction(1, rate)
        for p in s.encode(fr):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    return path


def test_audio_offset_files_recover_known_delay(tmp_path: Path):
    src = _clicks(6.0)
    d = int(0.0875 * SR)  # the sound reaches recording B 87.5 ms later on B's timeline
    noise = np.random.default_rng(1)
    a = src + 0.02 * noise.standard_normal(len(src))
    b = np.concatenate([np.zeros(d), src])[: len(src)] + 0.02 * noise.standard_normal(len(src))
    fa, fb = _write_wav(tmp_path / "a.wav", a), _write_wav(tmp_path / "b.wav", b)
    r = audio_offset(str(fa), str(fb), sample_rate=SR)
    assert r.status == "OK"
    assert r.offset_ms == pytest.approx(-87.5, abs=0.2)  # t_a = t_b + offset
    assert r.uncertainty_ms is not None and r.uncertainty_ms < 1.0
    assert r.n_windows >= 3 and any("mux offset" in w for w in r.warnings)


def test_audio_offset_resampled_input_and_sub_sample(tmp_path: Path):
    src = _clicks(5.0, seed=3)
    # 44.1 kHz recordings resampled to 16 kHz; B leads by 12.34 ms
    t = np.arange(int(5.0 * 44100)) / 44100
    s44 = np.interp(t, np.arange(len(src)) / SR, src)
    s44b = np.interp(t + 0.01234, np.arange(len(src)) / SR, src)
    fa, fb = _write_wav(tmp_path / "a.wav", s44, 44100), _write_wav(tmp_path / "b.wav", s44b, 44100)
    r = audio_offset(str(fa), str(fb), sample_rate=SR)
    assert r.status == "OK" and r.offset_ms == pytest.approx(12.34, abs=0.2)


def test_audio_offset_no_audio_and_noise(tmp_path: Path):
    vid = write_video((np.zeros((64, 64, 3), np.uint8) for _ in range(10)), tmp_path / "v.mp4", 25)
    fa = _write_wav(tmp_path / "a.wav", _clicks(2.0))
    assert audio_offset(str(fa), str(vid)).status == "NO_AUDIO"
    rng = np.random.default_rng(5)
    r = audio_offset_from_signals(rng.standard_normal(SR * 4).astype(np.float32),
                                  rng.standard_normal(SR * 4).astype(np.float32), SR)
    assert r.status == "LOW_CONFIDENCE"


def test_event_offset_interval_contains_truth():
    true_off = 37.3  # t_a = t_b + 37.3
    fps_a, fps_b = 50.0, 100.0
    rng = np.random.default_rng(0)
    events_true_a = np.sort(rng.uniform(500, 5000, 5))
    ea = [np.ceil(t / (1000 / fps_a)) * (1000 / fps_a) for t in events_true_a]  # first frame showing the event
    eb = [np.ceil((t - true_off) / (1000 / fps_b)) * (1000 / fps_b) for t in events_true_a]
    r = event_offset(ea, eb, fps_a, fps_b)
    assert r.status == "OK"
    lo, hi = r.feasible_interval_ms
    assert lo <= true_off <= hi
    assert r.uncertainty_ms <= (1000 / fps_a + 1000 / fps_b) / 2 + 1e-9
    single = event_offset(ea[:1], eb[:1], fps_a, fps_b)
    assert single.uncertainty_ms >= r.uncertainty_ms


def test_event_offset_inconsistent_and_validation():
    r = event_offset([100.0, 500.0], [60.0, 300.0], 50, 50)  # implies offsets 40 and 200 ms
    assert r.status == "INCONSISTENT" and r.uncertainty_ms is None
    with pytest.raises(ValueError):
        event_offset([1.0], [1.0, 2.0], 50, 50)
    r = event_offset([(100.0, 120.0)], [(10.0, 15.0)])
    assert r.feasible_interval_ms == (85.0, 110.0)
    assert event_offset([], []).status == "INSUFFICIENT_DATA"


def test_apply_sync_sets_camera_offset():
    cam = look_at_camera("b", np.array([30.0, 10, 4]), np.array([0.0, 8, 0.5]), 2000, 1280, 720)
    s = SyncResult(method="visual_events", status="OK", offset_ms=12.5, uncertainty_ms=3.0)
    out = apply_sync(cam, s, reference_offset_ms=1.0, reference_uncertainty_ms=4.0)
    assert out.time_offset_ms == 13.5 and out.sync_uncertainty_ms == pytest.approx(5.0)
    assert cam.time_offset_ms == 0.0  # original untouched
    with pytest.raises(ValueError):
        apply_sync(cam, SyncResult(method="visual_events", status="INCONSISTENT", offset_ms=1.0))
