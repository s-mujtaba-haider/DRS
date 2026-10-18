"""Effect of camera synchronisation error on two-view 3D reconstruction and the stump-plane projection.

Synthetic only (known cameras, known trajectory): camera B's frames are taken
``sync_error`` ms later than its timestamps claim, the two views are
triangulated as if simultaneous, the flight model is fitted to the 3D points up
to the pad, and the projected ball position at the stump plane (y = 0) is
compared with the truth. Results are a geometric sensitivity study, not a
measurement of any real rig.

    python scripts/evaluate_sync_sensitivity.py [--out outputs/reports/sync_sensitivity.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.calibration.camera import look_at_camera  # noqa: E402
from ml.lbw.geometry import time_at_plane_y  # noqa: E402
from ml.reconstruction.triangulation import CameraObservation, associate_and_triangulate  # noqa: E402
from ml.trajectory.ballistic import DEFAULT_DRAG_C, FlightModel, InsufficientObservations, fit_to_points  # noqa: E402

BALL_R = 0.0364
SYNC_ERRORS_MS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def truth_model() -> FlightModel:
    # Release ~18.5 m from the striker's stumps at ~37.5 m/s (135 km/h), pitching ~5 m short of the stumps.
    theta = np.array([0.10, 18.5, 2.20, -0.25, -37.0, -4.0, -0.25, -33.0, 3.6])
    return FlightModel(theta, 0.0, True, BALL_R, 0.0, DEFAULT_DRAG_C)


def rig():
    a = look_at_camera("end_on", np.array([0.0, 42.0, 7.0]), np.array([0.0, 2.0, 0.4]), 4200.0, 1920, 1080)
    b = look_at_camera("side_on", np.array([32.0, 9.0, 4.0]), np.array([0.0, 8.0, 0.5]), 2600.0, 1920, 1080)
    for c in (a, b):
        c.validated = True
        c.sync_uncertainty_ms = 0.0
    return a, b


def run(fps: float, sync_ms: float, noise_px: float, seed: int, y_pad: float = 0.6) -> dict:
    m = truth_model()
    cam_a, cam_b = rig()
    rng = np.random.default_rng(seed)
    t_pad = time_at_plane_y(m, y_pad, 0.0)
    ts = np.arange(0.0, t_pad, 1.0 / fps)
    obs = []
    for k, t in enumerate(ts):
        for cam, dt in ((cam_a, 0.0), (cam_b, sync_ms / 1000.0)):
            X = m.position(np.array([t + dt]))[0]  # B actually exposes dt later than its timestamp says
            uv = cam.project(X[None])[0] + rng.normal(0, noise_px, 2) if noise_px else cam.project(X[None])[0]
            obs.append(CameraObservation(camera_id=cam.camera_id, frame_index=k, timestamp_ms=t * 1000.0,
                                         u=float(uv[0]), v=float(uv[1]), sigma_px=max(noise_px, 0.5)))
    rec = associate_and_triangulate([cam_a, cam_b], obs, sync_tolerance_ms=1.0, max_reprojection_px=1e6)
    t_rec = np.array([p.timestamp_ms / 1000.0 for p in rec.points])
    X_rec = np.array([p.xyz for p in rec.points])
    err3d = np.linalg.norm(X_rec - m.position(t_rec), axis=1)
    reproj = np.array([max(p.reprojection_errors_px.values()) for p in rec.points])
    out = {"n_points": len(rec.points), "pos_err_median_m": float(np.median(err3d)), "pos_err_max_m": float(err3d.max()),
           "reproj_max_px": float(reproj.max())}
    try:
        fit = fit_to_points(t_rec, X_rec, np.full(len(t_rec), 0.01), ball_radius=BALL_R)
        fm = fit.model
        fm_truth_t = time_at_plane_y(m, 0.0, 0.0)
        t_fit = time_at_plane_y(fm, 0.0, float(t_rec[0]))
        if t_fit is None:
            out["stump_plane"] = None
        else:
            p_fit, p_true = fm.position(np.array([t_fit]))[0], m.position(np.array([fm_truth_t]))[0]
            out["stump_plane"] = {"dx_m": float(p_fit[0] - p_true[0]), "dz_m": float(p_fit[2] - p_true[2]),
                                  "err_m": float(np.hypot(p_fit[0] - p_true[0], p_fit[2] - p_true[2]))}
        out["fit_status"] = fit.status
        out["fit_rms_m"] = fit.residual_rms
    except InsufficientObservations as exc:
        out["stump_plane"], out["fit_status"] = None, f"insufficient: {exc}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/reports/sync_sensitivity.json")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    rows = []
    for fps in (50.0, 100.0, 250.0):
        for noise in (0.0, 1.0):
            for s_ms in SYNC_ERRORS_MS:
                runs = [run(fps, s_ms, noise, seed) for seed in range(1 if noise == 0 else args.seeds)]
                sp = [r["stump_plane"]["err_m"] for r in runs if r.get("stump_plane")]
                row = {"fps": fps, "noise_px": noise, "sync_error_ms": s_ms, "n_runs": len(runs),
                       "n_points_median": float(np.median([r["n_points"] for r in runs])),
                       "pos_err_median_m": float(np.median([r["pos_err_median_m"] for r in runs])),
                       "pos_err_max_m": float(np.max([r["pos_err_max_m"] for r in runs])),
                       "reproj_max_px": float(np.max([r["reproj_max_px"] for r in runs])),
                       "stump_plane_err_median_m": float(np.median(sp)) if sp else None,
                       "stump_plane_err_p90_m": float(np.percentile(sp, 90)) if sp else None,
                       "stump_plane_projection_failures": len(runs) - len(sp)}
                rows.append(row)
                print(f"fps {fps:5.0f} noise {noise:.0f}px sync {s_ms:5.1f} ms: 3D median {row['pos_err_median_m'] * 100:6.2f} cm, "
                      f"max {row['pos_err_max_m'] * 100:6.2f} cm, reproj max {row['reproj_max_px']:6.2f} px, stump-plane median "
                      f"{(row['stump_plane_err_median_m'] or float('nan')) * 100:6.2f} cm p90 {(row['stump_plane_err_p90_m'] or float('nan')) * 100:6.2f} cm")
    report = {"description": "Synthetic two-camera sync-error sensitivity (end-on 4200 px + side-on 2600 px focal, 1920x1080); "
                             "ball ~37 m/s, observed until 0.6 m before the stumps then projected to the stump plane.",
              "synthetic": True, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
