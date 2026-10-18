"""Synthetic evaluation of multi-view LBW (ml.lbw.multiview) under camera synchronisation error.

Two calibrated cameras (end-on 50 fps, side-on 100 fps with a 3 ms phase offset, so exposures never
coincide) observe deliveries with known outcomes (truth model includes aerodynamic drag). Camera B
really exposes ``sync_error`` ms later than its timestamps claim. Scenarios:

* hw_0ms        hardware sync, no error, declared uncertainty 0
* sw_2ms        software sync, 2 ms actual error, 2 ms declared
* sw_20ms       software sync, 20 ms actual error, 20 ms declared (typical USB webcam rig)
* hw_2ms_undeclared  hardware sync claimed, 2 ms actual error, 0 declared (what an unnoticed error does)

Synthetic only: says nothing about real cameras, detectors or calibration.

    python scripts/evaluate_multiview_lbw.py [--n 60] [--out outputs/reports/multiview_lbw_synth.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.calibration.camera import look_at_camera  # noqa: E402
from ml.calibration.pitch_geometry import BallSpec, WicketSpec  # noqa: E402
from ml.evaluation.synthetic import make_delivery  # noqa: E402
from ml.lbw.engine import LbwConfig  # noqa: E402
from ml.lbw.geometry import time_at_plane_y  # noqa: E402
from ml.lbw.multiview import ViewObservations, analyze_lbw_multiview  # noqa: E402
from ml.lbw.rules import LbwFacts  # noqa: E402
from ml.trajectory.ballistic import DEFAULT_DRAG_C  # noqa: E402

SCENARIOS = {
    "hw_0ms": dict(err=0.0, declared=0.0, hw=True),
    "sw_2ms": dict(err=2.0, declared=2.0, hw=False),
    "sw_20ms": dict(err=20.0, declared=20.0, hw=False),
    "hw_2ms_undeclared": dict(err=2.0, declared=0.0, hw=True),
}


def rig():
    a = look_at_camera("end_on", np.array([0.0, 42.0, 7.0]), np.array([0.0, 2.0, 0.4]), 4200.0, 1920, 1080)
    b = look_at_camera("side_on", np.array([32.0, 9.0, 4.0]), np.array([0.0, 8.0, 0.5]), 2600.0, 1920, 1080)
    for c in (a, b):
        c.validated = True
    return a, b


def truth_hit(model, impact_t: float) -> bool:
    w, r = WicketSpec(), BallSpec().radius_m
    tc = time_at_plane_y(model, 0.0, impact_t)
    if tc is None:
        return False
    x, _, z = model.position(np.array([tc]))[0]
    dx = max(abs(x) - w.half_width_m, 0.0)
    dz = max(z - w.top_m, 0.0) + max(-z, 0.0)
    return bool(np.hypot(dx, dz) <= r)


def delivery(i: int, rng):
    d = make_delivery(pitch_y=float(rng.uniform(3.0, 7.0)), line_x=float(rng.uniform(-0.25, 0.25)),
                      end_x=float(rng.uniform(-0.35, 0.35)), speed_mps=float(rng.uniform(30, 40)),
                      post_bounce_vz=float(rng.uniform(1.5, 4.0)), fps=1000)
    model = replace(d.model, drag_c=DEFAULT_DRAG_C)  # truth with drag
    ts = np.arange(0, 1.5, 1e-4)
    impact_t = float(ts[np.argmax(model.position(ts)[:, 1] <= 1.2)])
    return model, impact_t


def observe(cam, model, fps, phase_s, t_end, err_s, noise_px, rng):
    t_claim = np.arange(phase_s, t_end, 1.0 / fps)
    X = model.position(t_claim + err_s)  # exposes err_s later than claimed
    uv = cam.project(X) + rng.normal(0, noise_px, (len(t_claim), 2))
    return t_claim * 1000.0, uv


def run(n: int, noise_px: float = 1.0, seed: int = 0) -> dict:
    cam_a, cam_b = rig()
    cfg = LbwConfig(n_samples=800)
    out = {}
    for name, sc in SCENARIOS.items():
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            model, impact_t = delivery(i, rng)
            truth = truth_hit(model, impact_t)
            ta, uva = observe(cam_a, model, 50.0, 0.0, impact_t, 0.0, noise_px, rng)
            tb, uvb = observe(cam_b, model, 100.0, 0.003, impact_t, sc["err"] / 1000.0, noise_px, rng)
            views = [
                ViewObservations(camera=cam_a, t_ms=ta.tolist(), uv=uva.tolist(), sigma_px=max(noise_px, 0.5),
                                 sync_uncertainty_ms=0.0, hardware_sync=sc["hw"]),
                ViewObservations(camera=cam_b, t_ms=tb.tolist(), uv=uvb.tolist(), sigma_px=max(noise_px, 0.5),
                                 sync_uncertainty_ms=sc["declared"], hardware_sync=sc["hw"]),
            ]
            res, diag, _ = analyze_lbw_multiview(views, LbwFacts(batter_handedness="right"), cfg,
                                                 impact_time_ms=impact_t * 1000.0)
            wp = res.wicket_projection
            err = None
            if wp and wp.nominal_crossing_xz:
                tc = time_at_plane_y(model, 0.0, impact_t)
                if tc is not None:
                    p = model.position(np.array([tc]))[0]
                    err = float(np.hypot(wp.nominal_crossing_xz[0] - p[0], wp.nominal_crossing_xz[1] - p[2]))
            rows.append({"truth_hit": truth, "verdict": res.verdict.value, "p_hit": res.p_hit, "stump_plane_err_m": err,
                         "sync_definite_allowed": diag.sync_definite_allowed if diag else None})
        definite = [r for r in rows if r["verdict"] in ("PROJECTED_TO_HIT", "PROJECTED_TO_MISS")]
        correct = [r for r in definite if (r["verdict"] == "PROJECTED_TO_HIT") == r["truth_hit"]]
        errs = [r["stump_plane_err_m"] for r in rows if r["stump_plane_err_m"] is not None]
        hits = [r for r in rows if r["truth_hit"]]
        out[name] = {
            **sc, "n": len(rows), "n_truth_hit": len(hits),
            "n_definite": len(definite), "n_correct_definite": len(correct), "n_confident_errors": len(definite) - len(correct),
            "inconclusive_rate": float(np.mean([r["verdict"] == "INCONCLUSIVE" for r in rows])),
            "insufficient_rate": float(np.mean([r["verdict"] == "INSUFFICIENT_DATA" for r in rows])),
            "hits_called_hit": sum(r["verdict"] == "PROJECTED_TO_HIT" for r in hits),
            "stump_plane_err_median_m": float(np.median(errs)) if errs else None,
            "stump_plane_err_p90_m": float(np.percentile(errs, 90)) if errs else None,
            "rows": rows,
        }
        s = out[name]
        print(f"{name:20s} n={s['n']} definite={s['n_definite']} correct={s['n_correct_definite']} "
              f"confident_errors={s['n_confident_errors']} inconclusive={s['inconclusive_rate']:.2f} "
              f"hits {s['hits_called_hit']}/{s['n_truth_hit']} stump-plane median "
              f"{(s['stump_plane_err_median_m'] or float('nan')) * 100:.1f} cm p90 {(s['stump_plane_err_p90_m'] or float('nan')) * 100:.1f} cm")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="outputs/reports/multiview_lbw_synth.json")
    args = ap.parse_args()
    res = run(args.n)
    report = {"description": __doc__.split("\n\n")[1], "synthetic": True, "noise_px": 1.0,
              "cameras": "end_on 50 fps (4200 px focal), side_on 100 fps (2600 px focal), 1920x1080, exact calibration",
              "scenarios": res}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
