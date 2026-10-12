"""Synthetic benchmark for crease registration from ONE detected wicket (near stumps hidden).

Random behind-the-bowler cameras render a synthetic frame (ml/evaluation/synth_pitch.py: white
creases on a pitch strip, an umpire hiding the near stumps, pixel noise); the far stumps box gets
1 px Gaussian edge noise. Registration (ml/calibration/pitch_registration.py) then estimates the
camera, and ground points near the striker are back-projected to measure the error.

    python scripts/evaluate_pitch_registration.py --n 40 --out outputs/reports/pitch_registration_synth.json

Reported for all cameras and for those passing the acceptance rule in ml/calibration/auto.py.
Synthetic frames are far cleaner than broadcast footage; this measures the geometry and the
optimiser, not real-world accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.calibration.auto import MIN_INLIER_RATIO, MIN_NEAR_END_INLIERS  # noqa: E402
from ml.calibration.camera import look_at_camera  # noqa: E402
from ml.calibration.pitch_geometry import PitchSpec, WicketSpec  # noqa: E402
from ml.calibration.pitch_registration import register_pitch  # noqa: E402
from ml.evaluation.synth_pitch import render  # noqa: E402
from scripts.evaluate_stumps_calibration import backproject, wicket_box  # noqa: E402


def stats(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())} if len(v) else {"n": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out")
    a = ap.parse_args()
    L, wk = PitchSpec().stumps_distance_m, WicketSpec()
    rng = np.random.default_rng(a.seed)
    ground = np.array([[x, y, 0.0] for x in np.linspace(-1.5, 1.5, 4) for y in np.linspace(0.5, 10, 5)])
    rows = []
    for i in range(a.n):
        d, h = rng.uniform(50, 110), rng.uniform(8, 20)
        f = rng.uniform(80, 130) * (d + L) / wk.top_m
        cam = look_at_camera("synth", np.array([rng.uniform(-2, 2), L + d, h]), np.array([0, rng.uniform(6, 12), 0.5]), f, 1920, 1080)
        img = render(cam, noise=4, seed=i)
        far = [*(np.array(wicket_box(cam, 0.0, wk)) + rng.normal(0, 1, 4)), 0.9]
        reg = register_pitch(img, [far])
        if reg.camera is None:
            rows.append({"failed": True})
            continue
        g = [np.linalg.norm(backproject(reg.camera, cam.project(X[None])[0], 2, 0.0) - X) for X in ground]
        rows.append({"failed": False, "focal_rel_err": float(reg.camera.K_np[0, 0] / f - 1), "ground_err_m": float(np.median(g)),
                     "inlier_ratio": reg.inlier_ratio, "bowler_end_inliers": reg.near_end_inliers,
                     "accepted": bool(reg.inlier_ratio >= MIN_INLIER_RATIO and reg.near_end_inliers >= MIN_NEAR_END_INLIERS)})
        print(i, {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    ok = [r for r in rows if not r["failed"]]
    acc = [r for r in ok if r["accepted"]]
    res = {"benchmark": "crease registration from one wicket, synthetic frames", "cameras": a.n,
           "failures": sum(r["failed"] for r in rows),
           "acceptance_rule": {"min_inlier_ratio": MIN_INLIER_RATIO, "min_bowler_end_inliers": MIN_NEAR_END_INLIERS},
           "all": {"ground_err_m": stats([r["ground_err_m"] for r in ok]), "abs_focal_rel_err": stats([abs(r["focal_rel_err"]) for r in ok])},
           "accepted": {"ground_err_m": stats([r["ground_err_m"] for r in acc]), "abs_focal_rel_err": stats([abs(r["focal_rel_err"]) for r in acc])},
           "caveat": "Clean synthetic frames; real footage adds clutter, worn and occluded creases, and detector error."}
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
