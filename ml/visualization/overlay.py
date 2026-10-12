"""Overlay rendering for annotated review videos.

Visual language (kept consistent with the web UI):
* observed detection  - solid green box + dot
* interpolated        - hollow amber circle
* smoothed trail      - cyan polyline
* fitted / predicted  - magenta
* other candidates    - thin grey boxes
"""

from __future__ import annotations

import cv2
import numpy as np

from ml.common.schemas import Detection, TrackingResult

GREEN = (80, 220, 80)
AMBER = (0, 190, 255)
CYAN = (255, 220, 0)
MAGENTA = (255, 0, 255)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)


class OverlayRenderer:
    def __init__(self, tracking: TrackingResult, detections: list[Detection], model_id: str, trail: int = 30,
                 fitted_path: list[tuple[float, float]] | None = None):
        self.dets_by_frame: dict[int, list[Detection]] = {}
        for d in detections:
            self.dets_by_frame.setdefault(d.frame_index, []).append(d)
        self.primary = next((t for t in tracking.tracks if t.is_primary), None)
        self.points = {}
        if self.primary:
            for p in self.primary.points:
                self.points.setdefault(p.frame_index, {})[p.kind] = p
        self.frames_sorted = sorted(self.points)
        self.model_id = model_id
        self.trail = trail
        self.fitted = fitted_path or []

    def draw(self, img: np.ndarray, frame_index: int, timestamp_ms: float) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        s = max(1.0, w / 1280)
        for d in self.dets_by_frame.get(frame_index, []):
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), GREY, 1)
        if self.primary:
            past = [f for f in self.frames_sorted if frame_index - self.trail <= f <= frame_index]
            pts = []
            for f in past:
                kinds = self.points[f]
                p = kinds.get("filtered") or kinds.get("interpolated")
                if p:
                    pts.append((int(p.x), int(p.y)))
            if len(pts) >= 2:
                cv2.polylines(out, [np.array(pts, np.int32)], False, CYAN, max(1, int(2 * s)), cv2.LINE_AA)
            cur = self.points.get(frame_index, {})
            if "observed" in cur:
                p = cur["observed"]
                det = next((d for d in self.dets_by_frame.get(frame_index, [])
                            if abs(d.center[0] - p.x) < 1e-3 and abs(d.center[1] - p.y) < 1e-3), None)
                if det:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    pad = int(4 * s)
                    cv2.rectangle(out, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), GREEN, max(1, int(2 * s)))
                cv2.circle(out, (int(p.x), int(p.y)), max(2, int(3 * s)), GREEN, -1, cv2.LINE_AA)
                label = f"T{self.primary.track_id} {p.confidence:.2f} {p.detection_source}"
                cv2.putText(out, label, (int(p.x) + int(10 * s), int(p.y) - int(10 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5 * s, GREEN, max(1, int(s)), cv2.LINE_AA)
            elif "interpolated" in cur:
                p = cur["interpolated"]
                cv2.circle(out, (int(p.x), int(p.y)), max(5, int(8 * s)), AMBER, max(1, int(2 * s)), cv2.LINE_AA)
                cv2.putText(out, f"T{self.primary.track_id} interpolated", (int(p.x) + int(10 * s), int(p.y) - int(10 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, AMBER, max(1, int(s)), cv2.LINE_AA)
        if len(self.fitted) >= 2:
            cv2.polylines(out, [np.array(self.fitted, np.int32)], False, MAGENTA, 1, cv2.LINE_AA)
        # Header bar
        cv2.rectangle(out, (0, 0), (w, int(28 * s)), (20, 20, 20), -1)
        hdr = f"frame {frame_index}  t={timestamp_ms / 1000:.3f}s  model={self.model_id}  EXPERIMENTAL - not an official decision"
        cv2.putText(out, hdr, (int(8 * s), int(19 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, WHITE, max(1, int(s)), cv2.LINE_AA)
        return out
