"""Multi-camera capture manager and clip export on the shared host clock.

Typical match-day use: start every camera at the start of play, keep a rolling
buffer of the last few minutes on disk, and when an appeal is referred export
the window around the delivery for every camera as review clips.

Two execution modes:

* ``process_per_camera=False`` (default): each camera is a pair of threads in
  this process. Simple; enough for one or two moderate-rate cameras.
* ``process_per_camera=True``: each camera records in its own process, so
  encoders do not compete for the Python GIL. ``time.perf_counter_ns`` is a
  system-wide clock (QueryPerformanceCounter / CLOCK_MONOTONIC), so the shared
  clock still holds across processes. Status and preview frames are exchanged
  through small files in the camera directory.

Export reads only from disk (segments plus their commit-log sidecars), so it
works identically in both modes, while a segment is still being written, and
after a crash. Exported clips are MP4 files whose frame PTS equal the host
arrival time minus the clip's first frame, so ``ml.video.ingest.iter_frames``
reproduces those times. Synchronisation between cameras is *software*
(shared arrival clock), never hardware-triggered: the unknown, camera-specific
exposure-to-arrival latency is not removed.
"""

from __future__ import annotations

import io
import json
import logging
import multiprocessing as mp
import os
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from ml.capture.recorder import RAW_FORMAT, CameraRecorder, RecorderConfig, SegmentInfo, read_sidecar, scan_segments
from ml.capture.sources import FrameSource, clock_ns, host_to_utc_ns, make_source

log = logging.getLogger("drs.capture")

TIMING_SCHEMA = "drs.capture.clip_timing/1"
_STATUS_FILE, _PREVIEW_FILE = "status.json", "preview.jpg"


def _jpeg(img: np.ndarray, quality: int = 80, max_width: int | None = 960) -> bytes | None:
    if max_width and img.shape[1] > max_width:
        h = int(round(img.shape[0] * max_width / img.shape[1]))
        img = cv2.resize(img, (max_width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # reader has it open (Windows); try again shortly
            time.sleep(0.01)
    tmp.unlink(missing_ok=True)


# ------------------------------------------------------------ process mode
def _camera_process_main(spec: dict, directory: str, cfg: dict, stop_ev, ready_ev, go_ev, preview_every_s: float) -> None:
    d = Path(directory)
    rec = None
    try:
        rec = CameraRecorder(make_source(spec), d, RecorderConfig(**cfg))
        rec.prepare()
    except Exception as exc:  # noqa: BLE001 - reported through the status file
        _atomic_write(d / _STATUS_FILE, json.dumps({"camera_id": spec.get("camera_id"), "running": False,
                                                    "error": f"start: {type(exc).__name__}: {exc}"}).encode())
        ready_ev.set()
        return
    ready_ev.set()
    go_ev.wait()
    rec.start()
    last_preview = 0.0
    try:
        while not stop_ev.is_set() and rec.is_alive():
            st = rec.status()
            st["pid"] = os.getpid()
            _atomic_write(d / _STATUS_FILE, json.dumps(st).encode())
            now = time.monotonic()
            if now - last_preview >= preview_every_s and (latest := rec.latest_frame()) is not None:
                jpg = _jpeg(latest[1])
                if jpg:
                    _atomic_write(d / _PREVIEW_FILE, jpg)
                last_preview = now
            stop_ev.wait(0.25)
    finally:
        rec.stop()
        st = rec.status()
        st["pid"] = os.getpid()
        _atomic_write(d / _STATUS_FILE, json.dumps(st).encode())


class ProcessCamera:
    """Parent-side handle of a camera recording in a child process."""

    def __init__(self, spec: dict, directory: Path, cfg: RecorderConfig, ctx, go_ev, preview_every_s: float = 0.25):
        if not isinstance(spec, dict):
            raise TypeError("process_per_camera requires dict source specs (see make_source)")
        self.spec, self.dir, self.cfg = dict(spec), Path(directory), cfg
        self.camera_id = spec["camera_id"]
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / _STATUS_FILE).unlink(missing_ok=True)
        self.stop_ev, self.ready_ev = ctx.Event(), ctx.Event()
        self.proc = ctx.Process(target=_camera_process_main, name=f"drs-cam-{self.camera_id}", daemon=True,
                                args=(self.spec, str(self.dir), cfg.to_dict(), self.stop_ev, self.ready_ev, go_ev,
                                      preview_every_s))

    @property
    def running(self) -> bool:
        return self.proc.is_alive()

    def _status_file(self) -> dict:
        try:
            return json.loads((self.dir / _STATUS_FILE).read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def status(self) -> dict:
        st = self._status_file()
        st.setdefault("camera_id", self.camera_id)
        st.setdefault("error", None)
        st["running"] = self.proc.is_alive() and st.get("running", True)
        st["process"] = {"pid": self.proc.pid, "alive": self.proc.is_alive(), "exitcode": self.proc.exitcode}
        if not self.proc.is_alive() and self.proc.exitcode not in (0, None) and not st["error"]:
            st["error"] = f"camera process exited with code {self.proc.exitcode}"
        return st

    def export_meta(self) -> dict:
        st = self._status_file()
        return {"nominal_fps": st.get("nominal_fps"), "description": st.get("source", self.camera_id),
                "codec": st.get("encoder"), "jitter_ms": st.get("arrival_jitter_ms")}

    def latest_jpeg(self) -> bytes | None:
        try:
            return (self.dir / _PREVIEW_FILE).read_bytes()
        except OSError:
            return None

    def segments_overlapping(self, t0_ns: int, t1_ns: int) -> list[SegmentInfo]:
        return [s for s in scan_segments(self.dir) if s.last_host_ns >= t0_ns and s.first_host_ns <= t1_ns]

    def stop(self, timeout: float = 15.0) -> None:
        self.stop_ev.set()
        self.proc.join(timeout)
        if self.proc.is_alive():
            log.error("camera %s did not stop; terminating", self.camera_id)
            self.proc.terminate()
            self.proc.join(5)

    def wait_finished(self, timeout: float | None = None) -> None:
        self.proc.join(timeout)


def _meta(rec) -> dict:
    if hasattr(rec, "export_meta"):
        return rec.export_meta()
    return {"nominal_fps": rec.source.nominal_fps, "description": rec.source.description, "codec": rec.codec,
            "jitter_ms": rec.arrival_jitter_ms()}


# ------------------------------------------------------------------ manager
class CaptureManager:
    def __init__(self, root: str | Path, cfg: RecorderConfig | None = None, process_per_camera: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or RecorderConfig()
        self.process_per_camera = process_per_camera
        self.recorders: dict[str, CameraRecorder | ProcessCamera] = {}
        self.audio = None  # optional ml.capture.audio.AudioRecorder on the same host clock
        self._lock = threading.Lock()
        self._ctx = mp.get_context("spawn") if process_per_camera else None

    # ------------------------------------------------------------------ control
    def _register(self, rec) -> None:
        with self._lock:
            old = self.recorders.get(rec.camera_id)
            if old is not None and old.running:
                raise ValueError(f"Camera {rec.camera_id!r} is already recording")
            self.recorders[rec.camera_id] = rec

    def add_camera(self, source: FrameSource | dict, cfg: RecorderConfig | None = None, start: bool = True):
        return self.start_cameras([source], cfg)[0] if start else self._create(source, cfg)

    def _create(self, source, cfg, go_ev=None):
        cfg = cfg or self.cfg
        if self.process_per_camera:
            cid = source["camera_id"] if isinstance(source, dict) else source.camera_id
            rec = ProcessCamera(source, self.root / cid, cfg, self._ctx, go_ev)
        else:
            src = make_source(source) if isinstance(source, dict) else source
            rec = CameraRecorder(src, self.root / src.camera_id, cfg)
        self._register(rec)
        return rec

    def start_cameras(self, sources: list[FrameSource | dict], cfg: RecorderConfig | None = None, ready_timeout_s: float = 60.0):
        """Prepare every camera (open sources and encoders) first, then start them together."""
        if self.process_per_camera:
            go = self._ctx.Event()
            recs = [self._create(s, cfg, go) for s in sources]
            for r in recs:
                r.proc.start()
            deadline = time.monotonic() + ready_timeout_s
            for r in recs:
                r.ready_ev.wait(max(0.0, deadline - time.monotonic()))
            go.set()
            return recs
        recs = [self._create(s, cfg) for s in sources]
        for r in recs:
            r.prepare()
        for r in recs:
            r.start()
        return recs

    def stop_camera(self, camera_id: str) -> None:
        rec = self.recorders.get(camera_id)
        if rec is None:
            raise KeyError(camera_id)
        rec.stop()

    def start_audio(self, source, segment_s: float = 10.0) -> None:
        """Record a microphone (e.g. stump mic) on the shared host clock alongside the cameras."""
        from ml.capture.audio import AudioRecorder

        if self.audio is not None and self.audio.running:
            raise ValueError("Audio is already recording")
        self.audio = AudioRecorder(source, self.root / "_audio", segment_s=segment_s, retention_s=self.cfg.retention_s)
        self.audio.start()

    def stop_all(self) -> None:
        for rec in list(self.recorders.values()):
            rec.stop()
        if self.audio is not None:
            self.audio.stop()

    def now_ns(self) -> int:
        return clock_ns()

    def status(self) -> dict:
        return {"clock": "host_perf_counter_ns", "now_host_ns": clock_ns(), "now_utc_ns": host_to_utc_ns(clock_ns()),
                "sync_method": "software_shared_clock", "hardware_sync": False,
                "mode": "process_per_camera" if self.process_per_camera else "threads",
                "cameras": [r.status() for r in self.recorders.values()],
                "audio": self.audio.status() if self.audio is not None else None}

    def latest_frame(self, camera_id: str, jpeg_quality: int = 80, max_width: int | None = 960) -> bytes | None:
        """JPEG of the most recent captured frame (preview only; independent of the encoder)."""
        rec = self.recorders.get(camera_id)
        if rec is None:
            raise KeyError(camera_id)
        if isinstance(rec, ProcessCamera):
            return rec.latest_jpeg()
        latest = rec.latest_frame()
        return None if latest is None else _jpeg(latest[1], jpeg_quality, max_width)

    # ------------------------------------------------------------------- export
    def export_last(self, seconds: float, out_dir: str | Path, camera_ids: list[str] | None = None, **kw) -> dict:
        t1 = clock_ns()
        return self.export_clip(t1 - int(seconds * 1e9), t1, out_dir, camera_ids, **kw)

    def export_clip(self, t_start_ns: int, t_end_ns: int, out_dir: str | Path, camera_ids: list[str] | None = None,
                    codec: str = "libx264", crf: int = 12) -> dict:
        """Export [t_start_ns, t_end_ns] (shared host clock) for each camera.

        Returns {"window", "cameras": {camera_id: {...}}, "sync", "warnings"}. Each camera entry gives
        ``clip_path``, ``timing_path``, ``n_frames``, ``gaps``, ``missing_frames``, ``time_offset_ms``
        (add to the clip's timestamps to get milliseconds since ``t_start_ns`` on the common clock,
        i.e. a value for ``CameraModel.time_offset_ms``) and ``sync_uncertainty_ms``.
        """
        if t_end_ns <= t_start_ns:
            raise ValueError("t_end_ns must be after t_start_ns")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ids = camera_ids or list(self.recorders)
        result = {"window": {"t_start_host_ns": t_start_ns, "t_end_host_ns": t_end_ns,
                             "t_start_utc_ns": host_to_utc_ns(t_start_ns), "duration_s": (t_end_ns - t_start_ns) / 1e9},
                  "cameras": {}, "warnings": []}
        for cid in ids:
            rec = self.recorders.get(cid)
            if rec is None:
                result["warnings"].append(f"Unknown camera {cid!r}")
                continue
            entry = export_from_disk(rec.dir, cid, _meta(rec), t_start_ns, t_end_ns, out, codec=codec, crf=crf)
            result["cameras"][cid] = entry
            if entry.get("status") != "ok":
                result["warnings"].append(f"{cid}: {entry.get('status')}")
            elif entry["missing_frames"]:
                result["warnings"].append(f"{cid}: {entry['missing_frames']} frame(s) missing in the window")
        if self.audio is not None:
            from ml.capture.audio import export_audio

            a = export_audio(self.audio.dir, t_start_ns, t_end_ns, out / "audio.wav", self.audio.source.sample_rate)
            if a.get("status") == "ok":
                a["offset_ms"] = (a["start_host_ns"] - t_start_ns) / 1e6  # first sample, ms after the window start
            result["audio"] = a
        unc = [c["sync_uncertainty_ms"] for c in result["cameras"].values() if c.get("sync_uncertainty_ms") is not None]
        result["sync"] = {"method": "software_shared_clock", "hardware_sync": False,
                          "max_sync_uncertainty_ms": max(unc) if unc else None,
                          "note": "Cameras share one host clock, but each camera's exposure-to-arrival latency is unknown and "
                                  "not compensated. Treat cross-camera timing as approximate unless verified against a "
                                  "common visual or audio event."}
        (out / "export.json").write_text(json.dumps(result, indent=2))
        return result


def decode_segment(seg: SegmentInfo):
    """Yield (record, BGR image) per decodable frame. Frames map to sidecar records by order.

    Only committed bytes are decoded (the last record's ``byte_end``), so a segment still being
    written, or truncated by a crash, yields its complete prefix.
    """
    recs = read_sidecar(seg.sidecar)
    if not recs:
        return
    try:
        with seg.path.open("rb") as fh:
            src = io.BytesIO(fh.read(recs[-1]["byte_end"]))
        container = av.open(src, format=RAW_FORMAT.get(seg.path.suffix, "h264"))
    except (OSError, av.error.FFmpegError):
        log.warning("segment %s unreadable", seg.path.name)
        return
    i = 0
    try:
        for fr in container.decode(container.streams.video[0]):
            if i >= len(recs):
                break
            yield recs[i], fr.to_ndarray(format="bgr24")
            i += 1
    except av.error.FFmpegError:
        log.warning("segment %s truncated; using decodable frames only", seg.path.name)
    finally:
        container.close()


def export_from_disk(camera_dir: Path, cid: str, meta: dict, t0: int, t1: int, out: Path,
                     codec: str = "libx264", crf: int = 12) -> dict:
    """Export one camera's frames with host time in [t0, t1] from its segment directory."""
    camera_dir = Path(camera_dir)
    segs = [s for s in scan_segments(camera_dir) if s.last_host_ns >= t0 and s.first_host_ns <= t1]
    tag = uuid.uuid4().hex[:8]
    pins = [s.path.with_name(f"{s.path.stem}.{tag}.pin") for s in segs]
    for p in pins:  # keep the recorder's retention from deleting what we are reading
        p.touch()
    try:
        return _encode_clip(segs, cid, meta, t0, t1, out, codec, crf)
    finally:
        for p in pins:
            p.unlink(missing_ok=True)


def _encode_clip(segs, cid, meta, t0, t1, out: Path, codec: str, crf: int) -> dict:
    clip = out / f"{cid}.mp4"
    tb = Fraction(1, 90000)
    nominal = meta.get("nominal_fps")
    container = stream = None
    frames: list[dict] = []
    pts_list: list[int] = []
    gaps: list[dict] = []
    clip_t0: int | None = None
    last_seq = None
    try:
        for seg in sorted(segs, key=lambda s: s.first_host_ns):
            for r, img in decode_segment(seg):
                if not (t0 <= r["host_ns"] <= t1):
                    continue
                if stream is None:
                    container = av.open(str(clip), "w")
                    h, w = img.shape[:2]
                    stream = container.add_stream(codec, rate=Fraction(nominal).limit_denominator(1001) if nominal else 30)
                    stream.width, stream.height, stream.pix_fmt = w - (w % 2), h - (h % 2), "yuv420p"
                    stream.codec_context.time_base = tb
                    stream.time_base = tb
                    if codec == "libx264":
                        stream.options = {"crf": str(crf), "preset": "fast", "bf": "0", "movflags": "+faststart"}
                    clip_t0 = r["host_ns"]
                pts = int(round((r["host_ns"] - clip_t0) * 90000 / 1e9))
                if pts_list and pts <= pts_list[-1]:
                    pts = pts_list[-1] + 1
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img[: stream.height, : stream.width]), format="bgr24")
                vf.pts, vf.time_base = pts, tb
                for p in stream.encode(vf):
                    container.mux(p)
                if frames:
                    queue_missing = max(r["seq"] - last_seq - 1, 0) if last_seq is not None and r["seq"] >= 0 else 0
                    src_missing = r.get("source_missing_before") or 0
                    if queue_missing or src_missing:
                        cause = ("source+encoder_queue" if queue_missing and src_missing else
                                 "encoder_queue" if queue_missing else "source")
                        gaps.append({"after_clip_frame": len(frames) - 1,
                                     "gap_ms": (r["host_ns"] - frames[-1]["host_ns"]) / 1e6,
                                     "est_missing_frames": queue_missing + src_missing, "cause": cause})
                last_seq = r["seq"]
                pts_list.append(pts)
                frames.append({"clip_frame_index": len(frames), "clip_time_ms": (r["host_ns"] - clip_t0) / 1e6,
                               "host_ns": r["host_ns"], "wall_utc_ns": r.get("wall_utc_ns"), "seq": r["seq"],
                               "backend_ts_ms": r.get("backend_ts_ms"), "backend_frame": r.get("backend_frame")})
        if stream is not None:
            for p in stream.encode():
                container.mux(p)
    finally:
        if container is not None:
            container.close()
    if not frames:
        return {"camera_id": cid, "status": "no_frames_in_window", "n_frames": 0, "clip_path": None, "timing_path": None,
                "gaps": 0, "missing_frames": 0, "time_offset_ms": None, "sync_uncertainty_ms": None}
    host = np.array([f["host_ns"] for f in frames], dtype=np.int64)
    gap_after = {g["after_clip_frame"] for g in gaps}
    clean = np.array([d for i, d in enumerate(np.diff(host) / 1e6) if i not in gap_after])
    interval = float(np.median(clean)) if len(clean) else (1000.0 / nominal if nominal else None)
    jitter = float(np.std(clean - np.median(clean))) if len(clean) >= 3 else meta.get("jitter_ms")
    sync_unc = (interval or 0.0) + (jitter or 0.0)
    timing = {
        "schema": TIMING_SCHEMA, "camera_id": cid, "source": meta.get("description"), "clip_path": clip.name,
        "encoder": meta.get("codec"), "export_codec": codec, "clip_t0_host_ns": clip_t0,
        "clip_t0_utc_ns": host_to_utc_ns(clip_t0), "requested_window_host_ns": [t0, t1],
        "time_offset_ms": (clip_t0 - t0) / 1e6, "nominal_fps": nominal,
        "measured_fps": 1000.0 / interval if interval else None, "n_frames": len(frames), "frames": frames, "gaps": gaps,
        "sync": {"clock": "host_perf_counter_ns", "method": "software_shared_clock", "hardware_sync": False,
                 "frame_interval_ms": interval, "arrival_jitter_ms": jitter, "sync_uncertainty_ms": sync_unc,
                 "note": "Host arrival time, not exposure time. Uncertainty = one frame interval (exposure instant "
                         "unknown within the interval) + arrival jitter; camera-specific constant latency is not included."},
        "observation_kind": "original_frames",
    }
    tpath = out / f"{cid}.timing.json"
    tpath.write_text(json.dumps(timing, indent=1))
    return {"camera_id": cid, "status": "ok", "clip_path": str(clip), "timing_path": str(tpath), "n_frames": len(frames),
            "gaps": len(gaps), "missing_frames": int(sum(g["est_missing_frames"] for g in gaps)),
            "time_offset_ms": timing["time_offset_ms"], "sync_uncertainty_ms": sync_unc,
            "first_host_ns": int(host[0]), "last_host_ns": int(host[-1])}
