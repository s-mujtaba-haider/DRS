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
        self.model_id = "motion:3frame-diff-v1"
        self.diff_threshold = diff_threshold
        self.min_area, self.max_area = min_area, max_area
        self.max_aspect, self.min_fill = max_aspect, min_fill
        self.compensate = compensate_camera_motion
        self.max_candidates = max_candidates
        self.work_width = work_width
        self._history: deque[np.ndarray] = deque(maxlen=2)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def reset(self) -> None:
        self._history.clear()

    def _prep(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        scale = 1.0
        if img.shape[1] > self.work_width:
            scale = self.work_width / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0), scale

    def _align(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        if not self.compensate:
            return prev
        (dx, dy), response = cv2.phaseCorrelate(prev.astype(np.float32), cur.astype(np.float32))
        if response < 0.1 or (abs(dx) < 0.5 and abs(dy) < 0.5):
            return prev
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(prev, m, (prev.shape[1], prev.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    def detect_frame(self, img: np.ndarray, info: FrameInfo) -> list[Detection]:
        gray, scale = self._prep(img)
        if len(self._history) < 2:
            self._history.append(gray)
            return []
        p2, p1 = self._history[0], self._history[1]
        d1 = cv2.absdiff(gray, self._align(p1, gray))
        d2 = cv2.absdiff(gray, self._align(p2, gray))
        self._history.append(gray)
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
            strength = float(np.mean(d1[y : y + h, x : x + w])) / 255.0
            compact = 1.0 / aspect
            score = float(np.clip(0.5 * compact + 0.5 * min(1.0, strength * 4), 0.0, 1.0))
            box = np.array([x, y, x + w, y + h], dtype=np.float64) / scale
            dets.append((score, box))
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
