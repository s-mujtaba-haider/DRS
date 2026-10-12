"""Camera calibration CLI.

Intrinsics from chessboard images (validated intrinsics):
    python scripts/calibrate_cameras.py intrinsics --images "calib/*.jpg" --pattern 9x6 --square 0.025 --out cam_K.json

Pose from pitch landmarks (JSON list of {"name": landmark, "image": [u, v]}):
    python scripts/calibrate_cameras.py pose --points clicks.json --size 1920x1080 [--intrinsics cam_K.json] --out camera.json

Without --intrinsics the focal length is estimated under stated assumptions (approximate mode).
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
    a = ap.parse_args()

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


if __name__ == "__main__":
    sys.exit(main())
