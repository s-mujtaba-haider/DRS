"""Shot-boundary (camera cut) detection for broadcast footage.

Ball tracks must not be associated across a cut. Consecutive frames are
compared with HSV colour-histogram correlation on a downscaled image; a cut is
declared when correlation drops below a threshold. Gradual transitions (fades,
wipes, replays' graphics) may be missed or split; the result is a heuristic.
"""

from __future__ import annotations

import cv2
import numpy as np


class ShotCutDetector:
    def __init__(self, threshold: float = 0.6, width: int = 160):
        self.threshold = threshold
        self.width = width
        self._prev: np.ndarray | None = None

    def _hist(self, img: np.ndarray) -> np.ndarray:
        h = int(img.shape[0] * self.width / img.shape[1])
        small = cv2.resize(img, (self.width, max(h, 1)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def is_cut(self, img: np.ndarray) -> bool:
        """True when ``img`` starts a new shot relative to the previous frame."""
        h = self._hist(img)
        prev, self._prev = self._prev, h
        if prev is None:
            return False
        return float(cv2.compareHist(prev, h, cv2.HISTCMP_CORREL)) < self.threshold


def detect_shot_cuts(video: str, max_frames: int | None = None, threshold: float = 0.6) -> list[int]:
    """Frame indices that start a new shot."""
    det = ShotCutDetector(threshold=threshold)
    cap = cv2.VideoCapture(video)
    cuts, i = [], 0
    while max_frames is None or i < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if det.is_cut(img):
            cuts.append(i)
        i += 1
    cap.release()
    return cuts
