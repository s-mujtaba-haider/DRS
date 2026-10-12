"""Synthetic benchmark: how well do two stumps boxes alone calibrate a broadcast delivery camera?

Random behind-the-bowler cameras (40-100 m behind the bowler's stumps, 6-20 m high, far wicket
80-130 px tall) project exact wicket boxes; Gaussian noise is added to every box edge. Only cameras
that see both wickets in full are kept (otherwise no detection exists to calibrate from).

    python scripts/evaluate_stumps_calibration.py --n 300 --out outputs/reports/stumps_calibration_synth.json

Errors reported (metres): ground points near the striker's end back-projected onto z=0, and points
on the striker's stump plane (y=0, z 0.1-0.7 m) back-projected onto that plane - the two places
where single-camera DRS geometry matters most.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.calibration.camera import CalibrationError, look_at_camera  # noqa: E402
from ml.calibration.pitch_geometry import PitchSpec, WicketSpec  # noqa: E402
from ml.calibration.stumps_landmarks import calibrate_from_stumps  # noqa: E402

W, H = 1920, 1080


def wicket_box(cam, y0: float, wk: WicketSpec):
    r = wk.stump_diameter_m / 2
    pts = np.array([[sx * wk.half_width_m, y0 + dy, z] for sx in (-1, 1) for dy in (-r, r) for z in (0.0, wk.top_m)])
    uv = cam.project(pts)
    return [*uv.min(0), *uv.max(0)]


def backproject(cam, uv, axis: int, value: float) -> np.ndarray:
    o, d = cam.ray(np.array([uv]))
    return o + (value - o[axis]) / d[axis] * d


def run(n: int, sigmas: list[float], seed: int) -> dict:
    pitch, wk = PitchSpec(), WicketSpec()
    L = pitch.stumps_distance_m
    ground = np.array([[x, y, 0.0] for x in np.linspace(-1.5, 1.5, 4) for y in np.linspace(0.5, 10, 5)])
    plane = np.array([[x, 0.0, z] for x in np.linspace(-0.5, 0.5, 4) for z in (0.1, 0.4, 0.7)])
    results = {}
    for sigma in sigmas:
        rng = np.random.default_rng(seed)
        rows, attempted, failures = [], 0, 0
        while attempted < n:
            d, h, lat = rng.uniform(40, 100), rng.uniform(6, 20), rng.uniform(-2, 2)
            f = rng.uniform(80, 130) * (d + L) / wk.top_m
            cam = look_at_camera("synth", np.array([lat, L + d, h]), np.array([0.0, rng.uniform(4, 16), 0.5]), f, W, H)
            boxes = [wicket_box(cam, y0, wk) for y0 in (0.0, L)]
            if any(b[0] < 0 or b[1] < 0 or b[2] > W or b[3] > H for b in boxes):
                continue  # a wicket is out of frame: nothing to detect
            attempted += 1
            noisy = [[*(np.array(b) + rng.normal(0, sigma, 4)), 0.9] for b in boxes]
            try:
                est, _ = calibrate_from_stumps(noisy, (W, H))
            except CalibrationError:
                est = None
            if est is None:
                failures += 1
                continue
            g = [np.linalg.norm(backproject(est, cam.project(X[None])[0], 2, 0.0) - X) for X in ground]
            s = [np.linalg.norm(backproject(est, cam.project(X[None])[0], 1, 0.0) - X) for X in plane]
            rows.append({"focal_rel_err": abs(est.K_np[0, 0] / f - 1), "ground_err_m": float(np.median(g)),
                         "stump_plane_err_m": float(np.median(s)), "rms_px": est.reprojection_rms_px})

        def stat(k, rows=rows):
            v = np.array([r[k] for r in rows])
            return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())} if len(v) else None

        results[f"sigma_{sigma}px"] = {"cameras": attempted, "failures": failures,
                                       **{k: stat(k) for k in ("focal_rel_err", "ground_err_m", "stump_plane_err_m", "rms_px")}}
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sigmas", default="0,0.5,1,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {"benchmark": "stumps-box calibration, synthetic broadcast cameras", "image_size": [W, H],
           "results": run(a.n, [float(s) for s in a.sigmas.split(",")], a.seed),
           "caveat": "Exact synthetic geometry; real boxes add detector bias, occlusion and lens distortion."}
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
