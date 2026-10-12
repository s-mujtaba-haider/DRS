"""Camera calibration CLI.

Intrinsics from chessboard images (validated intrinsics):
    python scripts/calibrate_cameras.py intrinsics --images "calib/*.jpg" --pattern 9x6 --square 0.025 --out cam_K.json

Pose from pitch landmarks (JSON list of {"name": landmark, "image": [u, v]}):
    python scripts/calibrate_cameras.py pose --points clicks.json --size 1920x1080 [--intrinsics cam_K.json] --out camera.json

Without --intrinsics the focal length is estimated under stated assumptions (approximate mode).

Automatic seed from detected stumps (behind-the-bowler view, both wickets visible):
    python scripts/calibrate_cameras.py auto --video clip.mp4 [--frame 40] --out camera.json --overlay check.jpg

The automatic result is approximate and never validated; check the overlay and correct the
landmarks by hand (pose subcommand) before relying on it.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.calibration.camera import calibrate_intrinsics_chessboard, estimate_pose  # noqa: E402
from ml.calibration.pitch_geometry import standard_landmarks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("intrinsics")
    i.add_argument("--images", required=True)
    i.add_argument("--pattern", default="9x6", help="inner corners, e.g. 9x6")
    i.add_argument("--square", type=float, required=True, help="square size in metres")
    i.add_argument("--out", required=True)
    p = sub.add_parser("pose")
    p.add_argument("--points", required=True)
    p.add_argument("--size", required=True, help="WIDTHxHEIGHT")
    p.add_argument("--intrinsics")
    p.add_argument("--camera-id", default="main")
    p.add_argument("--out", required=True)
    sub.add_parser("landmarks")
    au = sub.add_parser("auto")
    au.add_argument("--video", required=True)
    au.add_argument("--frame", type=int, default=None, help="frame index; default: scan for the best two-wicket frame")
    au.add_argument("--scan-step", type=int, default=5)
    au.add_argument("--weights", default=str(ROOT / "models" / "checkpoints" / "cricket_stumps_best.pt"))
    au.add_argument("--imgsz", type=int, default=1280)
    au.add_argument("--conf", type=float, default=0.25)
    au.add_argument("--max-rms", type=float, default=3.0, help="reject calibrations above this reprojection RMS (px)")
    au.add_argument("--camera-id", default="main")
    au.add_argument("--out", required=True)
    au.add_argument("--overlay", help="write the frame with projected pitch markings for visual checking")
    a = ap.parse_args()

    if a.cmd == "auto":
        return auto_calibrate(a)

    if a.cmd == "landmarks":
        print(json.dumps(standard_landmarks(), indent=1))
        return 0
    if a.cmd == "intrinsics":
        files = sorted(glob.glob(a.images))
        imgs = [cv2.imread(f) for f in files]
        cols, rows = map(int, a.pattern.lower().split("x"))
        K, dist, rms, n = calibrate_intrinsics_chessboard([x for x in imgs if x is not None], (cols, rows), a.square)
        out = {"K": K.tolist(), "dist": dist.tolist(), "rms_px": rms, "views_used": n, "views_total": len(files), "validated": rms < 1.0}
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        if rms >= 1.0:
            print("WARNING: RMS >= 1 px; intrinsics not marked validated")
        return 0
    pts = json.loads(Path(a.points).read_text())
    lm = standard_landmarks()
    world = np.array([p.get("world") or lm[p["name"]] for p in pts], float)
    image = np.array([p["image"] for p in pts], float)
    w, h = map(int, a.size.lower().split("x"))
    K = dist = None
    validated = False
    if a.intrinsics:
        intr = json.loads(Path(a.intrinsics).read_text())
        K, dist, validated = np.array(intr["K"]), np.array(intr["dist"]), bool(intr.get("validated"))
    cam = estimate_pose(a.camera_id, world, image, (w, h), K=K, dist=dist, intrinsics_validated=validated)
    cam.save(a.out)
    print(cam.model_dump_json(indent=2))
    return 0


def _stumps_boxes(model, frame, imgsz: int, conf: float) -> list[list[float]]:
    r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    return [[*map(float, b), float(s)] for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def draw_pitch_overlay(frame: np.ndarray, cam, boxes) -> np.ndarray:
    """Projected stumps, creases and pitch edges (green) over the detected stumps boxes (red)."""
    from ml.calibration.pitch_geometry import PitchSpec, WicketSpec

    out = frame.copy()
    p, wk = PitchSpec(), WicketSpec()
    L, hw, bc, pc = p.stumps_distance_m, wk.half_width_m, p.bowling_crease_half_length_m, p.popping_crease_from_stumps_m
    segs = []
    for y0, s in ((0.0, 1.0), (L, -1.0)):
        segs += [((-bc, y0, 0), (bc, y0, 0)), ((-1.83, y0 + s * pc, 0), (1.83, y0 + s * pc, 0))]
        segs += [((x, y0, 0), (x, y0, wk.top_m)) for x in (-hw, 0.0, hw)]
        segs += [((-hw, y0, wk.top_m), (hw, y0, wk.top_m))]
    w2 = p.pitch_width_m / 2
    segs += [((-w2, 0, 0), (-w2, L, 0)), ((w2, 0, 0), (w2, L, 0))]
    for b in boxes:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
    for a_, b_ in segs:
        P = np.array([a_, b_], float)
        if not np.all(cam.is_in_front(P)):
            continue
        uv = cam.project(P)
        cv2.line(out, tuple(map(int, uv[0])), tuple(map(int, uv[1])), (0, 255, 0), 1, cv2.LINE_AA)
    return out


def auto_calibrate(a) -> int:
    from ultralytics import YOLO

    from ml.calibration.camera import CalibrationError
    from ml.calibration.stumps_landmarks import calibrate_from_stumps, landmarks_from_stumps

    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [a.frame] if a.frame is not None else list(range(0, n, a.scan_step))
    # Calibrate every frame with a consistent wicket pair and keep the lowest reprojection error;
    # without any, keep the best-scoring detections as manual-click seeds.
    best, seeds = None, None  # (rms, idx, frame, boxes, cam, lm), (score, idx, frame, boxes)
    for i in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        boxes = _stumps_boxes(model, frame, a.imgsz, a.conf)
        h, w = frame.shape[:2]
        try:
            cam, lm = calibrate_from_stumps(boxes, (w, h), camera_id=a.camera_id, min_score=a.conf)
        except CalibrationError:
            cam = None
        if cam is not None and cam.reprojection_rms_px <= a.max_rms and (best is None or cam.reprojection_rms_px < best[0]):
            best = (cam.reprojection_rms_px, i, frame, boxes, cam, lm)
        score = max((b[4] for b in boxes), default=-1.0)
        if seeds is None or score > seeds[0]:
            seeds = (score, i, frame, boxes)
    if best is None and seeds is None:
        print("ERROR: no readable frame")
        return 2
    if best:
        _, idx, frame, boxes, cam, lm = best
    else:
        _, idx, frame, boxes = seeds
        cam, lm = None, landmarks_from_stumps(boxes, min_score=a.conf)
        lm.warnings.append(f"No frame gave a two-wicket calibration with reprojection RMS <= {a.max_rms} px")
    report = {"video": a.video, "frame": idx, "stumps_boxes": boxes, "wickets_used": lm.wickets_used,
              "landmark_seeds": lm.points, "assumptions": lm.assumptions, "warnings": lm.warnings,
              "camera": cam.model_dump(mode="json") if cam else None}
    if cam:
        cam.save(a.out)
        report["camera_file"] = a.out
    Path(a.out).with_suffix(".auto.json").write_text(json.dumps(report, indent=1))
    if a.overlay:
        cv2.imwrite(a.overlay, draw_pitch_overlay(frame, cam, boxes) if cam else frame)
    print(json.dumps({k: v for k, v in report.items() if k not in ("landmark_seeds", "camera")}
                     | ({"focal_px": cam.K[0][0], "reprojection_rms_px": cam.reprojection_rms_px} if cam else {}), indent=1))
    return 0 if cam else 1


if __name__ == "__main__":
    sys.exit(main())
