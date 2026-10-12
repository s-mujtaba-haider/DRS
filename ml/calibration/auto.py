"""Automatic calibration seed for a broadcast delivery clip (stumps detector + crease registration).

Strategy (every candidate must pass the broadcast-camera plausibility check):
1. Scan frames with the stumps detector.
2. Frames with a consistent two-wicket pair give a stumps-only calibration (8 points); the
   lowest-RMS one is used.
3. Otherwise the best stumps detections are returned as landmark seeds for manual clicking.

Optional, off by default (``use_crease_registration``): register the painted creases at both ends
(``pitch_registration``), which works with one wicket. It is accurate on clean synthetic frames,
but on real broadcast frames the creases are often only 5-15 grey levels above the pitch and even
correct cameras matched only 13-33% of crease points, so it is not used unless asked for (see
docs/EVALUATION.md).

Nothing here is ever marked validated: it is a starting point for manual correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ml.calibration.camera import CalibrationError, CameraModel
from ml.calibration.pitch_registration import register_pitch
from ml.calibration.stumps_landmarks import calibrate_from_stumps, landmarks_from_stumps

# Acceptance for the optional crease registration. Synthetic benchmark
# (scripts/evaluate_pitch_registration.py): accepted registrations had about 10 cm median ground
# error near the striker; lower inlier ratios were often focal-length-ambiguous.
MIN_INLIER_RATIO = 0.85
MIN_NEAR_END_INLIERS = 40


@dataclass
class AutoCalibration:
    camera: CameraModel | None
    method: str  # "crease_registration" | "two_wicket_stumps" | "seeds_only"
    frame_index: int
    stumps_boxes: list
    landmark_seeds: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {"method": self.method, "frame": self.frame_index, "stumps_boxes": self.stumps_boxes,
                "landmark_seeds": self.landmark_seeds, "evidence": self.evidence, "warnings": self.warnings,
                "camera": self.camera.model_dump(mode="json") if self.camera else None}


def detect_stumps(model, frame: np.ndarray, imgsz: int = 1280, conf: float = 0.25) -> list[list[float]]:
    r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    return [[*map(float, b), float(s)] for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def auto_calibrate_video(video: str, model, frames: list[int] | None = None, scan_step: int = 5, imgsz: int = 1280,
                         conf: float = 0.25, max_rms: float = 3.0, register_top: int = 3,
                         camera_id: str = "main", use_crease_registration: bool = False) -> AutoCalibration:
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = frames if frames is not None else list(range(0, n, scan_step))
    two_wicket, scanned = None, []  # (rms, idx, boxes, cam, lm); (score, idx, frame, boxes)
    for i in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        boxes = detect_stumps(model, frame, imgsz, conf)
        h, w = frame.shape[:2]
        try:
            cam, lm = calibrate_from_stumps(boxes, (w, h), camera_id=camera_id, min_score=conf)
        except CalibrationError:
            cam = None
        if cam is not None and cam.reprojection_rms_px <= max_rms and (two_wicket is None or cam.reprojection_rms_px < two_wicket[0]):
            two_wicket = (cam.reprojection_rms_px, i, boxes, cam, lm)
        score = max((b[4] for b in boxes), default=-1.0)
        scanned.append((score, i, frame, boxes))
    cap.release()
    if not scanned:
        return AutoCalibration(None, "seeds_only", -1, [], warnings=["No readable frame"])

    best_reg = None
    for score, i, frame, boxes in sorted(scanned, key=lambda s: -s[0])[:register_top if use_crease_registration else 0]:
        if score < conf:
            continue
        reg = register_pitch(frame, boxes, camera_id=camera_id, min_score=conf)
        if reg.camera is None:
            continue
        ok = reg.inlier_ratio >= MIN_INLIER_RATIO and reg.near_end_inliers >= MIN_NEAR_END_INLIERS
        key = (ok, reg.inlier_ratio)
        if best_reg is None or key > best_reg[0]:
            best_reg = (key, i, boxes, reg)

    if best_reg and best_reg[0][0]:
        _, i, boxes, reg = best_reg
        ev = {"crease_inlier_ratio": reg.inlier_ratio, "visible_crease_points": reg.visible_points,
              "bowler_end_crease_inliers": reg.near_end_inliers, "crease_chamfer_px": reg.chamfer_px,
              "crease_recall_px": reg.recall_px, "stumps_rms_px": reg.stumps_rms_px, "wickets_used": reg.landmarks.wickets_used}
        return AutoCalibration(reg.camera, "crease_registration", i, boxes, reg.landmarks.points, ev, reg.camera.warnings)
    if two_wicket:
        rms, i, boxes, cam, lm = two_wicket
        warn = list(cam.warnings)
        if best_reg:
            warn.append(f"Crease registration not accepted (inlier ratio {best_reg[3].inlier_ratio:.2f}, "
                        f"bowler-end inliers {best_reg[3].near_end_inliers})")
        return AutoCalibration(cam, "two_wicket_stumps", i, boxes, lm.points, {"stumps_rms_px": rms, "wickets_used": 2}, warn)
    score, i, frame, boxes = max(scanned, key=lambda s: s[0])
    lm = landmarks_from_stumps(boxes, min_score=conf)
    warn = [*lm.warnings, "No frame gave an accepted crease registration or a two-wicket calibration"]
    if best_reg:
        warn.append(f"Best crease registration: inlier ratio {best_reg[3].inlier_ratio:.2f}, "
                    f"bowler-end inliers {best_reg[3].near_end_inliers} (thresholds {MIN_INLIER_RATIO}, {MIN_NEAR_END_INLIERS})")
    return AutoCalibration(None, "seeds_only", i, boxes, lm.points, {}, warn)


def landmark_points(res: AutoCalibration, image_size: tuple[int, int]) -> list[dict]:
    """Named standard landmarks (``pitch_geometry.standard_landmarks``) placed on the frame.

    With a camera, every landmark that projects inside the image is returned, so the user only
    corrects what is off. Without one, the stumps-box corners become the stump base/top landmarks
    (the box top is the bail top, 1.27 cm above the stump-top landmark: about 1 px at typical zoom).
    """
    from ml.calibration.pitch_geometry import standard_landmarks

    w, h = image_size
    lm = standard_landmarks()
    if res.camera is not None:
        names = list(lm)
        X = np.array([lm[n] for n in names], float)
        front = res.camera.is_in_front(X)
        uv = res.camera.project(X)
        return [{"name": n, "image": [float(u), float(v)]} for n, (u, v), ok in zip(names, uv, front)
                if ok and 0 <= u < w and 0 <= v < h]
    out = []
    for p in res.landmark_seeds:  # labels like "striker_off_rh_base" / "striker_off_rh_bail_top"
        end, side = p["label"].split("_")[0], p["label"].split("_")[1]
        kind = "base" if p["label"].endswith("_base") else "top"
        name = f"{end}_{side}_stump_{kind}_rh"
        if name in lm:
            out.append({"name": name, "image": [float(p["image"][0]), float(p["image"][1])]})
    return out
