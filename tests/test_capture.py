"""Live capture: shared-clock timing, ring buffer, dropped-frame accounting, clip export."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from ml.capture import CaptureManager, RecorderConfig, SyntheticSource, clock_ns, pick_encoder
from ml.capture import recorder as recmod
from ml.capture.manager import decode_segment
from ml.capture.recorder import scan_segments
from ml.video.ingest import iter_frames, probe_video

CFG = dict(encoder="libx264", segment_s=0.4)


def _clip_times(path: str) -> np.ndarray:
    return np.array([f.timestamp_ms for f in iter_frames(path)])


def test_offline_capture_export_timing_and_drops(tmp_path: Path):
    mgr = CaptureManager(tmp_path / "buf", RecorderConfig(**CFG))
    src = SyntheticSource("A", 320, 240, 50, duration_s=1.2, realtime=False, drop_seqs={20, 21})
    rec = mgr.add_camera(src)
    rec.wait_finished(30)
    st = rec.status()
    assert st["frames_captured"] == 58 and st["frames_encoded"] == 58
    assert st["source_dropped_est"] == 2 and st["queue_dropped"] == 0
    first, last = st["buffer_first_host_ns"], st["buffer_last_host_ns"]
    res = mgr.export_clip(first, last, tmp_path / "clip")
    e = res["cameras"]["A"]
    assert e["status"] == "ok" and e["n_frames"] == 58
    timing = json.loads(Path(e["timing_path"]).read_text())
    assert timing["sync"]["hardware_sync"] is False and timing["observation_kind"] == "original_frames"
    assert timing["gaps"] == [{"after_clip_frame": 19, "gap_ms": pytest.approx(60.0, abs=0.01), "est_missing_frames": 2,
                               "cause": "source"}]
    expected = np.array([f["clip_time_ms"] for f in timing["frames"]])
    got = _clip_times(e["clip_path"])
    assert len(got) == len(expected) and np.abs(got - expected).max() < 1.0
    md = probe_video(e["clip_path"])
    assert (md.width, md.height) == (320, 240)


def test_export_window_subset_and_offset(tmp_path: Path):
    mgr = CaptureManager(tmp_path / "buf", RecorderConfig(**CFG))
    rec = mgr.add_camera(SyntheticSource("A", 160, 120, 100, duration_s=1.0, realtime=False))
    rec.wait_finished(30)
    st = rec.status()
    t0 = st["buffer_first_host_ns"] + int(0.2e9) + 1  # just after a frame
    t1 = t0 + int(0.3e9)
    e = mgr.export_clip(t0, t1, tmp_path / "clip")["cameras"]["A"]
    assert e["n_frames"] == 30
    assert 0 < e["time_offset_ms"] <= 10.0  # first frame after the window start, within one interval
    assert e["first_host_ns"] >= t0 and e["last_host_ns"] <= t1


def test_live_capture_export_while_recording(tmp_path: Path):
    mgr = CaptureManager(tmp_path / "buf", RecorderConfig(**CFG))
    mgr.start_cameras([SyntheticSource("A", 320, 240, 50), SyntheticSource("B", 320, 240, 100)])
    try:
        time.sleep(1.3)
        jpg = mgr.latest_frame("A")
        assert jpg is not None and jpg[:2] == b"\xff\xd8"
        res = mgr.export_last(0.8, tmp_path / "clip")  # includes the still-open segment
        st = mgr.status()
    finally:
        mgr.stop_all()
    for cid, fps in (("A", 50), ("B", 100)):
        e = res["cameras"][cid]
        assert e["status"] == "ok"
        assert e["n_frames"] >= 0.6 * 0.8 * fps
        timing = json.loads(Path(e["timing_path"]).read_text())
        expected = np.array([f["clip_time_ms"] for f in timing["frames"]])
        assert np.abs(_clip_times(e["clip_path"]) - expected).max() < 1.0
    assert all(c["error"] is None for c in st["cameras"])
    assert res["sync"]["hardware_sync"] is False


def test_encoder_backlog_drops_are_counted(tmp_path: Path, monkeypatch):
    real = recmod._Encoder.encode

    def slow(self, img, pts=None, keyframe=False):
        if img is not None:
            time.sleep(0.03)
        return real(self, img, pts, keyframe)

    monkeypatch.setattr(recmod._Encoder, "encode", slow)
    mgr = CaptureManager(tmp_path / "buf", RecorderConfig(**CFG, queue_size=2))
    rec = mgr.add_camera(SyntheticSource("A", 160, 120, 100, duration_s=0.6))
    rec.wait_finished(30)
    st = rec.status()
    assert st["queue_dropped"] > 0
    assert st["frames_encoded"] + st["queue_dropped"] == st["frames_captured"]
    e = mgr.export_clip(st["buffer_first_host_ns"], st["buffer_last_host_ns"], tmp_path / "clip")["cameras"]["A"]
    timing = json.loads(Path(e["timing_path"]).read_text())
    assert any("encoder_queue" in g["cause"] for g in timing["gaps"])


def test_truncated_segment_is_readable(tmp_path: Path):
    d = tmp_path / "buf"
    mgr = CaptureManager(d, RecorderConfig(**CFG))
    rec = mgr.add_camera(SyntheticSource("A", 320, 240, 50, duration_s=0.4, realtime=False))
    rec.wait_finished(30)
    seg = scan_segments(d / "A")[0]
    data = seg.path.read_bytes()
    seg.path.write_bytes(data[: len(data) // 2])  # simulate a crash mid-segment
    seg.path.with_name(seg.path.stem + ".summary.json").unlink()
    seg2 = scan_segments(d / "A")[0]
    assert seg2.complete is False
    frames = list(decode_segment(seg2))
    assert 0 < len(frames) < seg.n_frames
    assert [r["seq"] for r, _ in frames] == list(range(len(frames)))


def test_retention_prunes_old_segments(tmp_path: Path):
    mgr = CaptureManager(tmp_path / "buf", RecorderConfig(encoder="libx264", segment_s=0.2, retention_s=0.5))
    rec = mgr.add_camera(SyntheticSource("A", 160, 120, 50, duration_s=2.0, realtime=False))
    rec.wait_finished(30)
    st = rec.status()
    assert st["buffer_span_s"] <= 0.5 + 0.2 + 0.05
    assert len(list((tmp_path / "buf" / "A").glob("seg_*.h264"))) == st["segments"]


def test_shared_clock_and_encoder_choice():
    a, b = clock_ns(), clock_ns()
    assert b >= a
    assert pick_encoder("libx264") == "libx264"
    assert pick_encoder("auto") in recmod.ENCODER_PREFERENCE


def test_process_per_camera_mode(tmp_path: Path):
    mgr = CaptureManager(tmp_path / "buf", RecorderConfig(**CFG), process_per_camera=True)
    mgr.start_cameras([{"camera_id": "P1", "type": "synthetic", "width": 320, "height": 240, "fps": 50},
                       {"camera_id": "P2", "type": "synthetic", "width": 320, "height": 240, "fps": 100}])
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            st = mgr.status()["cameras"]
            if all((c.get("frames_encoded") or 0) >= 60 for c in st):
                break
            time.sleep(0.2)
        res = mgr.export_last(0.5, tmp_path / "clip")
        jpg = mgr.latest_frame("P1")
    finally:
        mgr.stop_all()
    assert mgr.status()["mode"] == "process_per_camera"
    assert all(c["error"] is None for c in st), st
    assert jpg is not None and jpg[:2] == b"\xff\xd8"
    for cid in ("P1", "P2"):
        e = res["cameras"][cid]
        assert e["status"] == "ok" and e["n_frames"] > 10
        timing = json.loads(Path(e["timing_path"]).read_text())
        expected = np.array([f["clip_time_ms"] for f in timing["frames"]])
        assert np.abs(_clip_times(e["clip_path"]) - expected).max() < 1.0


class _StallingSource(SyntheticSource):
    """Delivers frame 15 late (host-side stall) without skipping any device frame."""

    def read(self):
        fr = super().read()
        if fr is not None and fr.backend_frame == 15:
            fr.host_ns += 60_000_000  # arrives 60 ms late, as after a scheduler stall
        return fr


def test_late_frame_with_contiguous_counter_is_not_a_drop(tmp_path: Path):
    mgr = CaptureManager(tmp_path / "buf", RecorderConfig(**CFG))
    rec = mgr.add_camera(_StallingSource("A", 160, 120, 50, duration_s=0.8, realtime=False))
    rec.wait_finished(30)
    st = rec.status()
    assert st["source_dropped_est"] == 0 and st["late_arrivals"] >= 1
