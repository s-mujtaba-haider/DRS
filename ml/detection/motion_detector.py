"""Classical motion-based ball *candidate* generator.

This is a heuristic, not a learned detector: it finds small, compact blobs that
moved relative to both of the two previous frames (three-frame differencing),
optionally after compensating global camera translation with phase correlation.
It produces many false positives (players, bat, shadows); the tracker's motion
model is what separates the ball from clutter. Scores are heuristic and are not
calibrated probabilities.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ml.common.schemas import Detection
from ml.detection.base import FrameInfo, make_detection


class MotionCandidateDetector:
    is_cricket_specific = False

    def __init__(
        self,
        diff_threshold: int = 18,
        min_area: int = 4,
        max_area: int = 600,
        max_aspect: float = 3.0,
        min_fill: float = 0.35,
        compensate_camera_motion: bool = True,
        max_candidates: int = 15,
        work_width: int = 1280,
    ):
        self.model_id = "motion:3frame-colordiff-v2"
        self.diff_threshold = diff_threshold
        self.min_area, self.max_area = min_area, max_area
        self.max_aspect, self.min_fill = max_aspect, min_fill
        self.compensate = compensate_camera_motion
        self.max_candidates = max_candidates
        self.work_width = work_width
        self._history: deque = deque(maxlen=2)
        # Longer colour history for a per-candidate median background (ghost suppression). It must
        # be long enough that a ball lingering in one place for a few frames is still a minority.
        self._bg: deque = deque(maxlen=15)
        self._camera_moved = False
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def reset(self) -> None:
        self._history.clear()
        self._bg.clear()

    def _prep(self, img: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], float]:
        """(blurred BGR, blurred gray) at the working resolution.

        Differencing uses colour: a red ball on green grass can have almost the same luminance as
        the grass (e.g. BGR (40,40,200) vs (60,120,50) differ by ~5 grey levels), so a grayscale
        difference misses it. Alignment still uses grayscale.
        """
        scale = 1.0
        if img.shape[1] > self.work_width:
            scale = self.work_width / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        color = cv2.GaussianBlur(img, (3, 3), 0)
        return (color, cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)), scale

    def _shift(self, prev_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray | None:
        if not self.compensate:
            return None
        (dx, dy), response = cv2.phaseCorrelate(prev_gray.astype(np.float32), cur_gray.astype(np.float32))
        if response < 0.1 or (abs(dx) < 0.5 and abs(dy) < 0.5):
            return None
        self._camera_moved = True
        return np.float32([[1, 0, dx], [0, 1, dy]])

    def _is_ghost(self, cur_color: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """A three-frame-difference blob where the current frame matches the recent background.

        When the ball moves less than its own diameter between frames (e.g. a ball travelling away
        from a camera behind the bowler, near the bounce), its two previous positions overlap and
        the AND of the two differences leaves a blob where the ball *was*. At a real ball position
        the current frame differs from the median of recent frames; at such a ghost it does not.
        Only applied with a static camera and enough history.
        """
        if self._camera_moved or len(self._bg) < 5:
            return False
        pad = 2
        y0, y1 = max(0, y - pad), min(cur_color.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(cur_color.shape[1], x + w + pad)
        bg = np.median(np.stack([b[y0:y1, x0:x1] for b in self._bg]), axis=0)
        d = np.abs(cur_color[y0:y1, x0:x1].astype(np.float32) - bg).max(axis=2)
        return float(np.percentile(d, 90)) < self.diff_threshold

    def _diff(self, cur, prev) -> np.ndarray:
        """Per-pixel maximum absolute channel difference after aligning ``prev`` to ``cur``."""
        m = self._shift(prev[1], cur[1])
        pc = prev[0] if m is None else cv2.warpAffine(prev[0], m, (prev[0].shape[1], prev[0].shape[0]), borderMode=cv2.BORDER_REPLICATE)
        return cv2.absdiff(cur[0], pc).max(axis=2)

    def detect_frame(self, img: np.ndarray, info: FrameInfo) -> list[Detection]:
        cur, scale = self._prep(img)
        self._camera_moved = False
        if len(self._history) < 2:
            self._history.append(cur)
            self._bg.append(cur[0])
            return []
        p2, p1 = self._history[0], self._history[1]
        d1 = self._diff(cur, p1)
        d2 = self._diff(cur, p2)
        self._history.append(cur)
        mask = cv2.bitwise_and(
            (d1 > self.diff_threshold).astype(np.uint8), (d2 > self.diff_threshold).astype(np.uint8)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dets: list[tuple[float, np.ndarray]] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (self.min_area <= area <= self.max_area):
                continue
            aspect = max(w, h) / max(1, min(w, h))
            fill = area / float(w * h)
            if aspect > self.max_aspect or fill < self.min_fill:
                continue
            if self._is_ghost(cur[0], x, y, w, h):
                continue
            strength = float(np.mean(d1[y : y + h, x : x + w])) / 255.0
            compact = 1.0 / aspect
            score = float(np.clip(0.5 * compact + 0.5 * min(1.0, strength * 4), 0.0, 1.0))
            box = np.array([x, y, x + w, y + h], dtype=np.float64) / scale
            dets.append((score, box))
        self._bg.append(cur[0])
        dets.sort(key=lambda t: -t[0])
        return [make_detection(info, b, s, self.model_id, "motion") for s, b in dets[: self.max_candidates]]

    def detect_batch(self, images: list[np.ndarray], infos: list[FrameInfo]) -> list[list[Detection]]:
        # Stateful: frames must be supplied in temporal order.
        return [self.detect_frame(img, info) for img, info in zip(images, infos)]


class HybridDetector:
    """Union of a learned detector and motion candidates, de-duplicated by distance."""

    def __init__(self, primary, motion: MotionCandidateDetector, merge_radius_px: float = 8.0,
                 motion_weight: float = 0.3):
        self.primary, self.motion = primary, motion
        self.merge_radius = merge_radius_px
        # Motion blobs are far noisier than learned detections; scale their heuristic score
        # so the tracker's cost and ranking prefer model detections when both exist.
        self.motion_weight = motion_weight
        self.model_id = f"hybrid[{primary.model_id}+{motion.model_id}]"
        self.is_cricket_specific = primary.is_cricket_specific

    def detect_batch(self, images, infos):
        a = self.primary.detect_batch(images, infos)
        b = self.motion.detect_batch(images, infos)
        merged = []
        for da, db in zip(a, b):
            out = list(da)
            for m in db:
                if all(np.hypot(m.center[0] - d.center[0], m.center[1] - d.center[1]) > self.merge_radius for d in da):
                    out.append(m.model_copy(update={"confidence": m.confidence * self.motion_weight}))
            merged.append(out)
        return merged
