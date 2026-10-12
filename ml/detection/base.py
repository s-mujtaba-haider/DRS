"""Detector interface and helpers shared by all ball detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ml.common.schemas import Detection, TimestampSource


@dataclass
class FrameInfo:
    index: int
    timestamp_ms: float
    timestamp_source: TimestampSource


class BallDetector(Protocol):
    model_id: str
    is_cricket_specific: bool

    def detect_batch(self, images: list[np.ndarray], infos: list[FrameInfo]) -> list[list[Detection]]: ...


class ModelUnavailableError(RuntimeError):
    """Raised when requested weights are missing or cannot be loaded."""


def make_detection(info: FrameInfo, box, conf: float, model_id: str, source: str, label: str = "ball") -> Detection:
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 - x1 < 1.0:
        x2 = x1 + 1.0
    if y2 - y1 < 1.0:
        y2 = y1 + 1.0
    return Detection(
        frame_index=info.index,
        timestamp_ms=info.timestamp_ms,
        timestamp_source=info.timestamp_source,
        bbox=(x1, y1, x2, y2),
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        confidence=float(min(max(conf, 0.0), 1.0)),
        model_id=model_id,
        source=source,
        label=label,
        # Rough 1-sigma localisation uncertainty: a quarter of the box size, at least 1 px.
        uncertainty_px=max(1.0, 0.25 * max(x2 - x1, y2 - y1)),
    )


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> list[int]:
    """Plain NumPy non-maximum suppression; returns kept indices sorted by score."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tile windows (x0, y0, x1, y1) covering the whole frame."""
    if tile >= width and tile >= height:
        return [(0, 0, width, height)]
    stride = max(1, int(tile * (1.0 - overlap)))

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        s = list(range(0, size - tile, stride))
        s.append(size - tile)
        return sorted(set(s))

    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in starts(height) for x in starts(width)]
