"""Microphone capture on the shared clock: block timing fit, export alignment, and transient timing."""

import time
from pathlib import Path

import pytest

from ml.capture import CaptureManager, RecorderConfig, SyntheticSource
from ml.capture.audio import AudioRecorder, SyntheticAudioSource, export_audio
from ml.contact_analysis.audio import analyze_video_audio


def test_export_places_clicks_on_host_clock(tmp_path: Path):
    src = SyntheticAudioSource(click_every_s=0.25, jitter_ms=2.0, duration_s=2.0)
    rec = AudioRecorder(src, tmp_path / "a", segment_s=0.7)
    rec.start()
    rec._thread.join(10)
    assert rec.stats.blocks > 50 and len(rec.stats.segments) >= 3
    clicks = src.click_host_ns
    t0, t1 = clicks[1] - 100_000_000, clicks[-1] + 50_000_000
    a = export_audio(tmp_path / "a", t0, t1, tmp_path / "out.wav", 48000)
    assert a["status"] == "ok" and abs(a["clock_ratio"] - 1) < 1e-4
    assert a["fit_residual_ms"] < 3.0  # 2 ms callback jitter is averaged out by the fit
    r = analyze_video_audio(tmp_path / "out.wav")
    found = sorted(tr.t_ms for tr in r.transients)
    expected = [(c - a["start_host_ns"]) / 1e6 for c in clicks if t0 <= c <= t1]
    assert len(found) == len(expected)
    assert max(abs(f - e) for f, e in zip(found, expected)) < 1.5  # fit removes per-block jitter


def test_manager_exports_audio_with_clip(tmp_path: Path):
    mgr = CaptureManager(tmp_path / "buf", RecorderConfig(encoder="libx264", segment_s=0.5))
    mgr.start_cameras([SyntheticSource("cam", 320, 240, 50)])
    mgr.start_audio(SyntheticAudioSource(click_every_s=0.3))
    time.sleep(2.0)
    st = mgr.status()
    assert st["audio"]["running"] and st["audio"]["blocks"] > 20
    exp = mgr.export_last(1.2, tmp_path / "clip")
    mgr.stop_all()
    a = exp["audio"]
    assert a["status"] == "ok" and Path(a["path"]).is_file()
    assert a["offset_ms"] == pytest.approx(0.0, abs=25.0)  # audio covers the window from its start
