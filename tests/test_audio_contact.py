"""Audio transient (edge candidate) analysis on synthetic signals and an encoded video with audio."""

from pathlib import Path

import av
import numpy as np
import pytest

from ml.contact_analysis.audio import analyze_audio_transients, analyze_video_audio

SR = 48000


def _signal(click_ms=(512.0,), dur_s=1.5, seed=0, speech=True):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * SR)) / SR
    x = 0.003 * rng.normal(size=t.size)
    if speech:  # slow, loud low-frequency sound: must not count as an impact
        x += 0.3 * np.sin(2 * np.pi * 220 * t) * np.clip(np.sin(np.pi * (t - 0.9) / 0.5), 0, None) * (t > 0.9) * (t < 1.4)
    for c in click_ms:
        k = int(c / 1000 * SR)
        n = int(0.004 * SR)
        x[k:k + n] += 0.5 * np.sin(2 * np.pi * 7000 * np.arange(n) / SR) * np.exp(-np.arange(n) / (0.0008 * SR))
    return x.astype(np.float32)


def test_detects_sharp_click_and_ignores_slow_sound():
    r = analyze_audio_transients(_signal(), SR, audio_start_ms=100.0, frame_times_ms=[(i, i * 20.0) for i in range(100)])
    assert r.status == "TRANSIENTS_FOUND" and len(r.transients) == 1
    tr = r.transients[0]
    assert tr.t_ms == pytest.approx(612.0, abs=1.0)  # 512 ms into audio that starts at 100 ms on the video axis
    assert tr.rise_ms <= 2.0 and tr.snr_db > 20
    assert tr.nearest_frame_index == 31
    assert any("mux offset" in w for w in r.warnings)


def test_no_transient_and_window():
    r = analyze_audio_transients(_signal(click_ms=()), SR, 0.0)
    assert r.status == "NO_TRANSIENT" and "does not rule out" in " ".join(r.reasons)
    two = analyze_audio_transients(_signal(click_ms=(300.0, 700.0)), SR, 0.0, window_ms=(600.0, 800.0))
    assert [round(t.t_ms) for t in two.transients] == [700]


def test_low_sample_rate_warns():
    x = _signal()[::6]  # 8 kHz
    r = analyze_audio_transients(x, 8000, 0.0)
    assert any("32 kHz" in w for w in r.warnings)


def write_av_clip(path: Path, click_ms: float, fps: int = 50, n: int = 60) -> Path:
    c = av.open(str(path), "w")
    vs = c.add_stream("mpeg4", rate=fps)
    vs.width, vs.height, vs.pix_fmt = 160, 120, "yuv420p"
    a = c.add_stream("aac", rate=SR, layout="mono")
    for i in range(n):
        img = np.full((120, 160, 3), (i * 4) % 255, np.uint8)
        for p in vs.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            c.mux(p)
    x = _signal(click_ms=(click_ms,), dur_s=n / fps, speech=False)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    step = 1024
    for k in range(0, len(pcm), step):
        fr = av.AudioFrame.from_ndarray(pcm[k:k + step].reshape(1, -1), format="s16", layout="mono")
        fr.sample_rate, fr.pts = SR, k
        for p in a.encode(fr):
            c.mux(p)
    for st in (vs, a):
        for p in st.encode(None):
            c.mux(p)
    c.close()
    return path


def test_video_file_with_audio(tmp_path):
    p = write_av_clip(tmp_path / "edge.mp4", click_ms=600.0)
    r = analyze_video_audio(p, center_ms=600.0, half_window_ms=300.0)
    assert r.status == "TRANSIENTS_FOUND"
    assert min(abs(t.t_ms - 600.0) for t in r.transients) < 30.0  # AAC priming delay shifts audio slightly
    assert r.sample_rate_hz == SR


def test_noise_blip_before_onset_does_not_inflate_rise_time():
    x = _signal(click_ms=(500.0,), speech=False)
    k = int(0.4935 * SR)  # a small blip 6.5 ms before the click (as on real broadcast audio)
    x[k:k + 30] += 0.02 * np.sin(2 * np.pi * 6000 * np.arange(30) / SR)
    r = analyze_audio_transients(x, SR, 0.0)
    assert any(abs(t.t_ms - 500.0) < 1.5 and t.rise_ms <= 2.0 for t in r.transients)
