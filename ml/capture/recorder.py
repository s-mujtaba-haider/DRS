"""Per-camera ring-buffer recorder: capture thread -> bounded queue -> encoder thread.

One persistent encoder per camera writes short raw elementary-stream segments
(H.264 Annex B, or MPEG-4 Part 2), each starting with a forced IDR frame so it
decodes on its own. A raw stream has no container index, so a file cut short
by a crash is still decodable up to its last complete packet.

Frame timing does not live in the bitstream. Each segment has a JSONL sidecar
that doubles as a commit log: a frame's record (host clock, wall clock,
backend timestamp, sequence number, ``byte_end``) is appended only after its
packet is on disk. Readers therefore decode at most ``byte_end`` bytes and map
decoded frames to records by order (no B-frames, so decode order equals
capture order), whether the segment is complete, still being written, or was
truncated by a crash. A small summary JSON is written when a segment closes.

The capture loop never blocks on the encoder for live sources: when the queue
is full the frame is discarded and counted (``queue_dropped``) and the gap is
visible in the sidecar as a jump in ``seq``.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import statistics
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from av.video.frame import PictureType

from ml.capture.sources import FrameSource, clock_ns

log = logging.getLogger("drs.capture")

PTS_HZ = 90000
_TIME_BASE = {"mpeg4": Fraction(1, 60000)}  # MPEG-4 Part 2 limits the time-base denominator to 65535
_EXT = {"h264_nvenc": ".h264", "libx264": ".h264", "mpeg4": ".m4v"}
RAW_FORMAT = {".h264": "h264", ".m4v": "m4v"}
ENCODER_PREFERENCE = ("h264_nvenc", "libx264", "mpeg4")
_ENCODER_OPTIONS = {
    # Near-lossless constant quality, no B-frames (decode order == capture order).
    # forced-idr: segment-start keyframes are IDR; repeat headers so every segment carries SPS/PPS.
    "h264_nvenc": {"preset": "p4", "tune": "hq", "rc": "constqp", "qp": "16", "bf": "0", "forced-idr": "1"},
    "libx264": {"preset": "superfast", "crf": "14", "bf": "0", "forced-idr": "1", "x264-params": "repeat-headers=1"},
    "mpeg4": {"qscale": "2", "bf": "0"},
}
_ENCODER_CHECK: dict[str, bool] = {}


def time_base_for(codec: str) -> Fraction:
    return _TIME_BASE.get(codec, Fraction(1, PTS_HZ))


class _Encoder:
    """Persistent encoder for one camera (re-opening nvenc per segment stalls all threads for ~0.1 s)."""

    def __init__(self, codec: str, options: dict, width: int, height: int, fps: float | None, gop: int | None):
        cc = av.CodecContext.create(codec, "w")
        cc.width, cc.height, cc.pix_fmt = width - (width % 2), height - (height % 2), "yuv420p"
        self.tb = time_base_for(codec)
        cc.time_base = self.tb
        if fps:
            cc.framerate = Fraction(fps).limit_denominator(1001)
        opts = dict(options)
        if gop:
            opts.setdefault("g", str(gop))
        cc.options = opts
        cc.open()
        self.cc = cc

    def encode(self, img: np.ndarray | None, pts: int | None = None, keyframe: bool = False) -> list:
        if img is None:
            return list(self.cc.encode(None))
        w, h = self.cc.width, self.cc.height
        # OpenCV's BGR->I420 conversion is faster than swscale here and releases the GIL.
        yuv = cv2.cvtColor(np.ascontiguousarray(img[:h, :w]), cv2.COLOR_BGR2YUV_I420)
        vf = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
        vf.pts, vf.time_base = pts, self.tb
        if keyframe:
            vf.pict_type = PictureType.I
        return list(self.cc.encode(vf))


def encoder_works(codec: str) -> bool:
    """True if the encoder exists in this PyAV/FFmpeg build *and* opens (nvenc needs a driver)."""
    if codec not in _ENCODER_CHECK:
        try:
            enc = _Encoder(codec, _ENCODER_OPTIONS.get(codec, {}), 256, 144, 25, None)
            enc.encode(np.zeros((144, 256, 3), np.uint8), 0, keyframe=True)
            enc.encode(None)
            _ENCODER_CHECK[codec] = True
        except Exception:  # noqa: BLE001 - any failure means "not usable here"
            _ENCODER_CHECK[codec] = False
    return _ENCODER_CHECK[codec]


def pick_encoder(preferred: str = "auto") -> str:
    order = ENCODER_PREFERENCE if preferred == "auto" else (preferred, *ENCODER_PREFERENCE)
    for c in order:
        if encoder_works(c):
            return c
    raise RuntimeError("No usable video encoder in this PyAV build")


@dataclass
class RecorderConfig:
    segment_s: float = 2.0
    retention_s: float = 120.0
    max_disk_mb: float = 4096.0
    queue_size: int = 64
    encoder: str = "auto"
    encoder_options: dict[str, str] | None = None
    gop_frames: int | None = None  # default: one keyframe per second of nominal rate
    drop_gap_factor: float = 1.5

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentInfo:
    camera_id: str
    path: Path
    sidecar: Path
    index: int
    first_host_ns: int
    last_host_ns: int
    n_frames: int
    size_bytes: int
    complete: bool
    encoder: str | None = None

    def to_json(self) -> dict:
        return {"camera_id": self.camera_id, "path": self.path.name, "sidecar": self.sidecar.name, "index": self.index,
                "first_host_ns": self.first_host_ns, "last_host_ns": self.last_host_ns, "n_frames": self.n_frames,
                "size_bytes": self.size_bytes, "complete": self.complete, "encoder": self.encoder}


def read_sidecar(path: Path) -> list[dict]:
    """Committed timing records of a segment; a truncated last line (crash mid-write) is ignored."""
    out = []
    if not path.exists():
        return out
    with path.open("r") as fh:
        for line in fh:
            if not line.endswith("\n"):
                break
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def summary_path(seg_path: Path) -> Path:
    return seg_path.with_name(seg_path.stem + ".summary.json")


def is_pinned(seg_path: Path) -> bool:
    return any(seg_path.parent.glob(seg_path.stem + ".*.pin"))


def scan_segments(camera_dir: Path) -> list[SegmentInfo]:
    """Segment index from disk: complete segments from their summary, open or crashed ones from the sidecar."""
    segs = []
    for data in sorted(p for p in Path(camera_dir).glob("seg_*") if p.suffix in RAW_FORMAT):
        side = data.with_suffix(".jsonl")
        summ = summary_path(data)
        if summ.exists():
            try:
                d = json.loads(summ.read_text())
                segs.append(SegmentInfo(d["camera_id"], data, side, d["index"], d["first_host_ns"], d["last_host_ns"],
                                        d["n_frames"], d["size_bytes"], True, d.get("encoder")))
                continue
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        recs = read_sidecar(side)
        if not recs:
            continue
        segs.append(SegmentInfo(recs[0].get("camera_id", Path(camera_dir).name), data, side, int(data.stem.split("_")[1]),
                                recs[0]["host_ns"], recs[-1]["host_ns"], len(recs), recs[-1]["byte_end"], False))
    return segs


class _Segment:
    """One raw elementary-stream segment file plus its commit-log sidecar."""

    def __init__(self, camera_id: str, directory: Path, index: int, codec: str, start_pts: int):
        self.camera_id, self.index, self.codec, self.start_pts = camera_id, index, codec, start_pts
        self.path = directory / f"seg_{index:06d}{_EXT.get(codec, '.h264')}"
        self.sidecar_path = directory / f"seg_{index:06d}.jsonl"
        self.data = self.path.open("wb", buffering=0)
        self.side = self.sidecar_path.open("w", buffering=1)
        self.info = SegmentInfo(camera_id, self.path, self.sidecar_path, index, 0, 0, 0, 0, False, codec)

    def commit(self, packet: bytes, rec: dict) -> None:
        self.data.write(packet)
        self.info.size_bytes += len(packet)
        rec = {**rec, "segment": self.index, "byte_end": self.info.size_bytes}
        self.side.write(json.dumps(rec) + "\n")
        if self.info.n_frames == 0:
            self.info.first_host_ns = rec["host_ns"]
        self.info.last_host_ns = rec["host_ns"]
        self.info.n_frames += 1

    def close(self) -> SegmentInfo:
        self.data.close()
        self.side.close()
        if self.info.n_frames == 0:
            self.path.unlink(missing_ok=True)
            self.sidecar_path.unlink(missing_ok=True)
            return self.info
        self.info.complete = True
        summary_path(self.path).write_text(json.dumps(self.info.to_json()))
        return self.info


@dataclass
class CameraStats:
    frames_captured: int = 0
    frames_encoded: int = 0
    source_dropped_est: int = 0
    late_arrivals: int = 0  # frames delivered > drop_gap_factor intervals late with no counter gap
    queue_dropped: int = 0
    encode_ms_total: float = 0.0
    intervals_ns: deque = field(default_factory=lambda: deque(maxlen=600))
    last_host_ns: int | None = None
    started_host_ns: int | None = None
    error: str | None = None


class CameraRecorder:
    """Records one source into ``directory`` until stopped (or until a finite source ends)."""

    def __init__(self, source: FrameSource, directory: Path, cfg: RecorderConfig | None = None):
        self.source, self.cfg = source, cfg or RecorderConfig()
        self.camera_id = source.camera_id
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.codec = pick_encoder(self.cfg.encoder)
        self.options = self.cfg.encoder_options or _ENCODER_OPTIONS.get(self.codec, {})
        self.q: queue.Queue = queue.Queue(maxsize=self.cfg.queue_size)
        self.stats = CameraStats()
        self._closed: list[SegmentInfo] = scan_segments(self.dir)
        self._next_index = max((s.index for s in self._closed), default=-1) + 1
        self._open: list[_Segment] = []  # segments whose packets may still arrive (oldest first)
        self._seg_lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: tuple[int, np.ndarray] | None = None
        self._threads: list[threading.Thread] = []
        self._prepared = False
        self._pre_encoder: _Encoder | None = None
        self.running = False

    # ------------------------------------------------------------------ control
    def prepare(self) -> None:
        """Open the source and encoder without capturing (lets several cameras start together)."""
        self.source.open()
        self._prepared = True
        self._pre_encoder = None
        if self.source.width and self.source.height:
            # Open the encoder before frames flow: an nvenc open holds the GIL for ~0.1 s.
            self._pre_encoder = self._make_encoder(self.source.width, self.source.height)

    def start(self) -> None:
        if not self._prepared:
            self.prepare()
        self._prepared = False
        self._stop.clear()
        self.running = True
        self.stats.started_host_ns = clock_ns()
        self._threads = [threading.Thread(target=self._capture_loop, name=f"cap-{self.camera_id}", daemon=True),
                         threading.Thread(target=self._writer_loop, name=f"enc-{self.camera_id}", daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        self.running = False
        try:
            self.source.close()
        except Exception:  # noqa: BLE001
            pass

    def wait_finished(self, timeout: float | None = None) -> None:
        """Wait for a finite source (file / synthetic with duration) to be fully recorded."""
        for t in self._threads:
            t.join(timeout)
        self.running = False

    def is_alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    # ------------------------------------------------------------------- loops
    def _make_encoder(self, w: int, h: int) -> _Encoder:
        nf = self.source.nominal_fps
        return _Encoder(self.codec, self.options, w, h, nf, self.cfg.gop_frames or (int(round(nf)) if nf else None))

    def _capture_loop(self) -> None:
        seq = 0
        live = getattr(self.source, "realtime", True) and not (
            getattr(self.source, "is_file", False) and not getattr(self.source, "realtime_file", True))
        nominal_ns = 1e9 / self.source.nominal_fps if self.source.nominal_fps else None
        last_backend = None
        raw_iv: deque = deque(maxlen=31)
        st = self.stats
        try:
            while not self._stop.is_set():
                fr = self.source.read()
                if fr is None:
                    break
                gap_ns = fr.host_ns - st.last_host_ns if st.last_host_ns is not None else None
                if gap_ns is not None:
                    raw_iv.append(gap_ns)
                # Reference interval: what the camera actually delivers (cameras often run below their
                # declared rate, e.g. auto-exposure in low light), falling back to the nominal rate.
                ref_ns = statistics.median(raw_iv) if len(raw_iv) >= 10 else nominal_ns
                missing = 0
                late = gap_ns is not None and ref_ns and gap_ns > (self.cfg.drop_gap_factor if len(raw_iv) >= 10 else 2.5) * ref_ns
                if fr.backend_frame is not None and last_backend is not None:
                    # A contiguous device counter is authoritative: a late frame is not a lost frame,
                    # but its arrival timestamp is delayed, so it is counted separately.
                    missing = max(0, fr.backend_frame - last_backend - 1)
                    if late and missing == 0:
                        st.late_arrivals += 1
                elif late:
                    missing = int(round(gap_ns / ref_ns)) - 1
                last_backend = fr.backend_frame
                if gap_ns is not None and missing == 0:
                    st.intervals_ns.append(gap_ns)
                st.source_dropped_est += missing
                st.last_host_ns = fr.host_ns
                st.frames_captured += 1
                self._latest = (seq, fr.image)
                rec = {"camera_id": self.camera_id, "seq": seq, "host_ns": fr.host_ns, "wall_utc_ns": fr.wall_utc_ns,
                       "backend_ts_ms": fr.backend_ts_ms, "backend_frame": fr.backend_frame,
                       "gap_before_ms": None if gap_ns is None else gap_ns / 1e6, "source_missing_before": missing}
                seq += 1
                if live:
                    try:
                        self.q.put_nowait((fr, rec))
                    except queue.Full:
                        st.queue_dropped += 1
                else:
                    while not self._stop.is_set():
                        try:
                            self.q.put((fr, rec), timeout=0.2)
                            break
                        except queue.Full:
                            continue
        except Exception as exc:  # noqa: BLE001 - reported through status()
            log.exception("capture %s failed", self.camera_id)
            st.error = f"capture: {type(exc).__name__}: {exc}"
        finally:
            # End-of-stream marker; never block forever if the encoder thread has died.
            while True:
                try:
                    self.q.put(None, timeout=0.5)
                    break
                except queue.Full:
                    if not any(t.is_alive() for t in self._threads[1:]):
                        break

    def _close_segment(self, seg: _Segment) -> None:
        info = seg.close()
        with self._seg_lock:
            self._open.remove(seg)
            if info.n_frames:
                self._closed.append(info)
        self._prune()

    def _route(self, packets: list, pending: dict[int, dict]) -> None:
        for pkt in packets:
            pts = pkt.pts
            target = max((g for g in self._open if g.start_pts <= pts), key=lambda g: g.start_pts, default=self._open[0])
            for g in [g for g in self._open if g.start_pts < target.start_pts]:
                self._close_segment(g)  # no B-frames: once a newer segment receives data, older ones are complete
            rec = pending.pop(pts, None)
            if rec is None:  # should not happen; keep the bitstream consistent anyway
                log.warning("%s: packet pts %s without a frame record", self.camera_id, pts)
                rec = {"camera_id": self.camera_id, "seq": -1, "host_ns": target.info.last_host_ns, "pts": pts}
            target.commit(bytes(pkt), rec)

    def _writer_loop(self) -> None:
        import time

        enc: _Encoder | None = None
        cur: _Segment | None = None
        cur_first = 0
        seg_ns = int(self.cfg.segment_s * 1e9)
        session_t0 = 0
        last_pts = -1
        pending: dict[int, dict] = {}
        try:
            while True:
                try:
                    item = self.q.get(timeout=0.05)
                except queue.Empty:
                    if self._stop.is_set() and self.q.empty():
                        break
                    continue
                if item is None:
                    break
                fr, rec = item
                t0 = time.perf_counter()
                if enc is None:
                    h, w = fr.image.shape[:2]
                    pre = self._pre_encoder
                    if pre is not None and (pre.cc.width, pre.cc.height) == (w - w % 2, h - h % 2):
                        enc = pre
                    else:
                        enc = self._make_encoder(w, h)
                    self._pre_encoder = None
                    session_t0 = fr.host_ns
                tb = enc.tb
                pts = int(round((fr.host_ns - session_t0) * tb.denominator / (1e9 * tb.numerator)))
                pts = max(pts, last_pts + 1)  # strictly increasing
                last_pts = pts
                new_seg = cur is None or fr.host_ns - cur_first >= seg_ns
                if new_seg:
                    cur = _Segment(self.camera_id, self.dir, self._next_index, self.codec, pts)
                    cur_first = fr.host_ns
                    self._next_index += 1
                    with self._seg_lock:
                        self._open.append(cur)
                pending[pts] = {**rec, "pts": pts}
                self._route(enc.encode(fr.image, pts, keyframe=new_seg), pending)
                self.stats.encode_ms_total += (time.perf_counter() - t0) * 1000
                self.stats.frames_encoded += 1
            if enc is not None:
                self._route(enc.encode(None), pending)
        except Exception as exc:  # noqa: BLE001
            log.exception("encoder %s failed", self.camera_id)
            self.stats.error = f"encoder: {type(exc).__name__}: {exc}"
            self._stop.set()
        finally:
            for g in list(self._open):
                try:
                    self._close_segment(g)
                except Exception as exc:  # noqa: BLE001
                    self.stats.error = self.stats.error or f"segment close: {exc}"

    # --------------------------------------------------------------- retention
    def _prune(self) -> None:
        """Delete the oldest closed segments beyond the retention time or disk budget (pinned ones are kept)."""
        with self._seg_lock:
            if not self._closed:
                return
            newest = max(s.last_host_ns for s in self._closed)
            total = sum(s.size_bytes for s in self._closed)
            max_bytes = self.cfg.max_disk_mb * 1e6
            keep = []
            for s in sorted(self._closed, key=lambda s: s.first_host_ns):
                expired = (newest - s.last_host_ns) / 1e9 > self.cfg.retention_s or total > max_bytes
                if expired and not is_pinned(s.path):
                    try:
                        for p in (s.path, s.sidecar, summary_path(s.path)):
                            p.unlink(missing_ok=True)
                        total -= s.size_bytes
                        continue
                    except PermissionError:  # open by a reader (Windows); retry on the next prune
                        pass
                keep.append(s)
            self._closed = keep

    def segments_overlapping(self, t0_ns: int, t1_ns: int) -> list[SegmentInfo]:
        return [s for s in scan_segments(self.dir) if s.last_host_ns >= t0_ns and s.first_host_ns <= t1_ns]

    # ------------------------------------------------------------------ status
    def latest_frame(self) -> tuple[int, np.ndarray] | None:
        return self._latest

    def arrival_jitter_ms(self) -> float | None:
        iv = list(self.stats.intervals_ns)
        if len(iv) < 5:
            return None
        return float(np.std(np.asarray(iv) - statistics.median(iv)) / 1e6)

    def measured_fps(self) -> float | None:
        iv = list(self.stats.intervals_ns)
        if len(iv) < 2:
            return None
        return 1e9 / float(np.mean(iv[-120:]))

    def _rate_mismatch(self) -> str | None:
        fps, nominal = self.measured_fps(), self.source.nominal_fps
        if fps and nominal and abs(fps - nominal) / nominal > 0.1:
            return f"camera delivers {fps:.1f} fps, declared {nominal:g} fps (check exposure/bandwidth settings)"
        return None

    def status(self) -> dict:
        st = self.stats
        segs = scan_segments(self.dir)
        disk = sum(s.size_bytes for s in segs)
        span = (min(s.first_host_ns for s in segs), max(s.last_host_ns for s in segs)) if segs else None
        now = clock_ns()
        return {
            "camera_id": self.camera_id, "source": self.source.description, "running": self.running and self.is_alive(),
            "nominal_fps": self.source.nominal_fps, "measured_fps": self.measured_fps(),
            "width": self.source.width, "height": self.source.height,
            "frames_captured": st.frames_captured, "frames_encoded": st.frames_encoded,
            "source_dropped_est": st.source_dropped_est, "late_arrivals": st.late_arrivals, "queue_dropped": st.queue_dropped,
            "encoder": self.codec, "encoder_backlog": self.q.qsize(),
            "encode_ms_per_frame": st.encode_ms_total / st.frames_encoded if st.frames_encoded else None,
            "arrival_jitter_ms": self.arrival_jitter_ms(),
            "rate_mismatch": self._rate_mismatch(),
            "last_frame_age_s": (now - st.last_host_ns) / 1e9 if st.last_host_ns else None,
            "segments": len(segs), "disk_bytes": disk,
            "buffer_span_s": (span[1] - span[0]) / 1e9 if span else 0.0,
            "buffer_first_host_ns": span[0] if span else None, "buffer_last_host_ns": span[1] if span else None,
            "disk_free_bytes": shutil.disk_usage(self.dir).free, "error": st.error,
        }
