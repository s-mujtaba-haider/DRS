"""Frame sources for live capture, all stamped on one shared process clock.

Every frame carries the host monotonic time (``time.perf_counter_ns``) taken
when the frame reached this process, plus wall-clock UTC and, where the
backend reports it, the backend's own timestamp and frame counter. The host
time is *arrival* time: the exposure-to-arrival latency of each camera is
unknown (USB/network buffering, encoder delay) and may differ between cameras.
That is software synchronisation, not hardware triggering.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

# One clock for every camera in the process. perf_counter_ns is monotonic and
# high resolution on Windows (QueryPerformanceCounter) and Linux (CLOCK_MONOTONIC).
clock_ns = time.perf_counter_ns
# Wall clock at a known point of the monotonic clock, to map host times to UTC.
CLOCK_ANCHOR = (clock_ns(), time.time_ns())


def host_to_utc_ns(host_ns: int) -> int:
    return CLOCK_ANCHOR[1] + (host_ns - CLOCK_ANCHOR[0])


@dataclass
class RawFrame:
    image: np.ndarray  # BGR uint8
    host_ns: int
    wall_utc_ns: int
    backend_ts_ms: float | None = None
    backend_frame: int | None = None


class FrameSource:
    """Base class. ``read`` blocks until the next frame and returns None at end of stream."""

    camera_id: str
    nominal_fps: float | None = None
    width: int | None = None
    height: int | None = None
    description: str = ""

    def open(self) -> None:  # pragma: no cover - interface
        pass

    def read(self) -> RawFrame | None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass


class OpenCVSource(FrameSource):
    """Device index, video file, or RTSP/HTTP URL via cv2.VideoCapture."""

    def __init__(self, camera_id: str, uri: int | str, width: int | None = None, height: int | None = None,
                 fps: float | None = None, backend: int | None = None, realtime_file: bool = True):
        self.camera_id, self.uri = camera_id, uri
        self.req_width, self.req_height, self.req_fps, self.backend = width, height, fps, backend
        self.is_file = isinstance(uri, str) and not uri.lower().startswith(("rtsp://", "http://", "https://", "udp://"))
        self.realtime_file = realtime_file
        self.cap = None
        self.description = f"opencv:{uri}"
        self._t0: int | None = None
        self._n = 0

    def open(self) -> None:
        import cv2

        uri = self.uri
        if isinstance(uri, str) and uri.isdigit():
            uri = int(uri)
        self.cap = cv2.VideoCapture(uri, self.backend) if self.backend is not None else cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise OSError(f"Cannot open capture source {self.uri!r}")
        if self.req_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        if self.req_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        if self.req_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.nominal_fps = float(fps) if fps and fps > 0 and math.isfinite(fps) else self.req_fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None

    def read(self) -> RawFrame | None:
        import cv2

        if self.cap is None:
            raise RuntimeError("source not opened")
        if self.is_file and self.realtime_file and self.nominal_fps:
            # Replay a file at its nominal rate so it behaves like a live feed.
            if self._t0 is None:
                self._t0 = clock_ns()
            _sleep_until(self._t0 + int(self._n * 1e9 / self.nominal_fps))
        ok, img = self.cap.read()
        host = clock_ns()
        if not ok or img is None:
            return None
        self._n += 1
        pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        pos_fr = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return RawFrame(img, host, time.time_ns(),
                        backend_ts_ms=float(pos_ms) if pos_ms and pos_ms > 0 else None,
                        backend_frame=int(pos_fr) if pos_fr and pos_fr > 0 else None)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _sleep_until(target_ns: int) -> None:
    while True:
        rem = target_ns - clock_ns()
        if rem <= 0:
            return
        # time.sleep is coarse on Windows; sleep most of the way, spin the rest.
        if rem > 2_000_000:
            time.sleep((rem - 1_500_000) / 1e9)


class SyntheticSource(FrameSource):
    """Deterministic test pattern: a white ball crossing a green field at a fixed rate.

    ``drop_every`` / ``drop_seqs`` skip frames (the time slot passes but nothing is
    emitted) to exercise dropped-frame detection. ``realtime`` paces frames on the
    shared clock; without it frames are produced as fast as possible with
    synthetic host times spaced at exactly 1/fps.
    """

    def __init__(self, camera_id: str, width: int = 1280, height: int = 720, fps: float = 50.0,
                 duration_s: float | None = None, realtime: bool = True, drop_seqs: set[int] | None = None,
                 drop_every: int | None = None, ball_radius_px: int = 8, texture: str = "flat"):
        self.camera_id, self.width, self.height, self.nominal_fps = camera_id, width, height, float(fps)
        self.duration_s, self.realtime = duration_s, realtime
        self.drop_seqs, self.drop_every = set(drop_seqs or ()), drop_every
        self.r = ball_radius_px
        if texture not in ("flat", "noise"):
            raise ValueError("texture must be 'flat' or 'noise'")
        self.texture = texture
        self.description = f"synthetic:{width}x{height}@{fps:g}" + (":noise" if texture == "noise" else "")
        self._slot = 0
        self._t0: int | None = None
        self._bg: np.ndarray | None = None

    def open(self) -> None:
        bg = np.zeros((self.height, self.width, 3), np.uint8)
        bg[:] = (40, 120, 40)
        bg[:, self.width // 2 - 40: self.width // 2 + 40] = (120, 170, 190)  # pitch strip
        if self.texture == "noise":
            # Temporal sensor-like noise (sigma ~4 levels, 8 fields cycled) so encoders see realistic entropy.
            rng = np.random.default_rng(0)
            self._bgs = [np.clip(bg.astype(np.int16) + rng.normal(0, 4, bg.shape).astype(np.int16), 0, 255).astype(np.uint8)
                         for _ in range(8)]
        else:
            self._bgs = [bg]
        self._bg = bg
        self._slot, self._t0 = 0, clock_ns()

    def ball_position(self, slot: int) -> tuple[int, int]:
        """Pixel position of the ball in time slot ``slot`` (known ground truth)."""
        period = max(int(self.nominal_fps * 2), 2)
        f = (slot % period) / period
        x = int(self.r + f * (self.width - 2 * self.r))
        y = int(self.height * (0.25 + 0.5 * abs(math.sin(math.pi * f))))
        return x, y

    def _dropped(self, slot: int) -> bool:
        return slot in self.drop_seqs or bool(self.drop_every and slot and slot % self.drop_every == 0)

    def read(self) -> RawFrame | None:
        import cv2

        if self._bg is None:
            raise RuntimeError("source not opened")
        while True:
            slot = self._slot
            self._slot += 1
            if self.duration_s is not None and slot >= self.duration_s * self.nominal_fps:
                return None
            due = self._t0 + int(round(slot * 1e9 / self.nominal_fps))
            if self._dropped(slot):
                continue
            break
        if self.realtime:
            _sleep_until(due)
            host = clock_ns()
        else:
            host = due
        img = self._bgs[slot % len(self._bgs)].copy()
        cv2.circle(img, self.ball_position(slot), self.r, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(img, f"{self.camera_id} #{slot}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return RawFrame(img, host, host_to_utc_ns(host), backend_ts_ms=slot * 1000.0 / self.nominal_fps,
                        backend_frame=slot)

    def close(self) -> None:
        self._bg = None


def make_source(spec: dict) -> FrameSource:
    """Build a source from a config dict: {camera_id, type: opencv|synthetic, uri, width, height, fps, ...}."""
    spec = dict(spec)
    kind = spec.pop("type", "opencv")
    cid = spec.pop("camera_id")
    if kind == "synthetic":
        return SyntheticSource(cid, **spec)
    if kind == "opencv":
        return OpenCVSource(cid, spec.pop("uri"), **spec)
    raise ValueError(f"Unknown source type {kind!r}")
