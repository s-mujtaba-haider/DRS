"""Physics-refinement study on cricket-synth (synthetic): which flight-model terms help, on held-out clips.

Three experiments, each comparing model variants on the same clips:

1. monocular   - the single-camera LBW benchmark (scripts/evaluate_trajectory_synth.py protocol):
                 fixed drag vs per-delivery drag (Cd prior), and a separate post-bounce lateral term.
2. points3d    - fit_to_points on ground-truth 3D positions + 1 cm noise (multi-view quality):
                 stump-plane error with drag only vs + swing vs + swing + lift vs + fitted drag.
3. multiview   - full multi-view LBW (ml.lbw.multiview) with the cricket-synth front-on camera and a
                 manifest side-on camera viewing the same simulated deliveries (exact projections,
                 1 px noise, hardware sync at 60 fps, the side camera with the best view of each
                 delivery): verdict accuracy and confirmed hits per variant.

Tuning used clips 0-119 of the front-on rig; final numbers use clips 240+ (never used before).
Synthetic data with exact cameras: not evidence of real-world accuracy.

    python scripts/evaluate_physics_refinement.py [--offset 240] [--limit 200] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data/raw/hf-cricket-synth/out/labels"
MONO_VARIANTS = {
    "fixed_drag (current default)": {},
    "fitted_drag Cd~N(0.4,0.1)": {"drag_cd_prior": (0.4, 0.1)},
    "separate_post_bounce_swing": {"swing_post_prior_sigma": 2.0},
    "fitted_drag + post_bounce_swing": {"drag_cd_prior": (0.4, 0.1), "swing_post_prior_sigma": 2.0},
}
PTS_VARIANTS = {
    "drag_only (multiview default before)": {},
    "+swing": {"swing_prior_sigma": 2.0},
    "+swing+lift": {"swing_prior_sigma": 2.0, "lift_prior_sigma": 3.0},
    "+swing+lift+fitted_drag": {"swing_prior_sigma": 2.0, "lift_prior_sigma": 3.0, "drag_cd_prior": (0.4, 0.1)},
}
MV_VARIANTS = {
    "drag_only, inflation 2 (previous default)": {"swing_prior_sigma": None, "lift_prior_sigma": None, "covariance_inflation": 2.0},
    "+swing+lift, inflation 2": {"swing_prior_sigma": 2.0, "lift_prior_sigma": 3.0, "covariance_inflation": 2.0},
    "+swing+lift, inflation 8 (calibrated on clips 0-119)": {"swing_prior_sigma": 2.0, "lift_prior_sigma": 3.0,
                                                             "covariance_inflation": 8.0},
}


def _front_files(offset: int, limit: int) -> list[Path]:
    out = []
    for f in sorted(DATA.glob("*.json")):
        d = json.loads(f.read_text())
        if d["camera"]["rig"] == "front_on_stump":
            out.append(f)
    return out[offset: offset + limit]


def _manifests() -> dict:
    m = {}
    for mf in (DATA.parents[1] / "manifests").glob("*.jsonl"):
        for line in mf.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                m[r["clip_id"]] = r
    return m


# ------------------------------------------------------------------ monocular
def _mono_one(args):
    path, variant, true_cd = args
    from ml.evaluation.synth_benchmark import evaluate_clip, load_clip
    from ml.lbw.engine import LbwConfig

    r = evaluate_clip(load_clip(path), impact_y=1.5, pixel_noise=1.0, lbw_cfg=LbwConfig(n_samples=500, covariance_inflation=2.0),
                      swing_sigma=2.0, fit_kwargs=MONO_VARIANTS[variant], true_cd=true_cd)
    return variant, r


def run_monocular(files, man, workers):
    from ml.evaluation.synth_benchmark import summarize

    jobs = [(f, v, man.get(json.loads(f.read_text())["clip_id"], {}).get("cd")) for v in MONO_VARIANTS for f in files]
    res: dict[str, list] = {v: [] for v in MONO_VARIANTS}
    with ProcessPoolExecutor(workers) as ex:
        for v, r in ex.map(_mono_one, jobs, chunksize=4):
            res[v].append(r)
    return {v: summarize(rs) for v, rs in res.items()}


# ------------------------------------------------------------------ 3D points
def _truth(d):
    from ml.evaluation.synth_benchmark import HALF_L, true_crossing

    fr = [x for x in d["frames"] if x["ball_in_scene"] and x["world_m"] is not None]
    t = np.array([x["t_s"] for x in fr])
    X = np.array([x["world_m"] for x in fr], float)
    X[:, 1] += HALF_L
    return t, X, true_crossing(t, X)


def run_points3d(files, noise=0.01, seed=1):
    from ml.lbw.geometry import time_at_plane_y
    from ml.trajectory.ballistic import DEFAULT_DRAG_C, InsufficientObservations, fit_to_points

    rng = np.random.default_rng(seed)
    errs = {v: [] for v in PTS_VARIANTS}
    for f in files:
        d = json.loads(f.read_text())
        t, X, tc = _truth(d)
        keep = X[:, 1] > 1.5
        if tc is None or keep.sum() < 8:
            continue
        tt, XX = t[keep], X[keep] + rng.normal(0, noise, X[keep].shape)
        for v, kw in PTS_VARIANTS.items():
            kw = {"drag_c": DEFAULT_DRAG_C, **kw} if "drag_cd_prior" not in kw else kw
            try:
                fit = fit_to_points(tt, XX, sigma_m=np.full(len(tt), noise), ball_radius=d["ball_radius_m"], **kw)
                ty = time_at_plane_y(fit.model, 0.0, float(tt[-1]))
            except (InsufficientObservations, ValueError):
                ty = None
            if ty is None:
                errs[v].append(None)
                continue
            p = fit.model.position(ty)[0]
            errs[v].append(float(math.hypot(p[0] - tc[1][0], p[2] - tc[1][2])))
    out = {}
    for v, e in errs.items():
        ok = np.array([x for x in e if x is not None])
        out[v] = {"n": len(e), "failed": sum(1 for x in e if x is None), "median_m": float(np.median(ok)),
                  "p90_m": float(np.percentile(ok, 90)), "max_m": float(ok.max())}
    return out


# ------------------------------------------------------------------ multi-view LBW
def _mv_one(args):
    path, side_rows, variant, sync_err_ms, inflation = args
    from ml.calibration.pitch_geometry import WicketSpec
    from ml.evaluation.synth_benchmark import camera_from_clip, manifest_camera
    from ml.lbw.engine import LbwConfig
    from ml.lbw.multiview import MultiViewConfig, ViewObservations, analyze_lbw_multiview
    from ml.lbw.rules import LbwFacts

    d = json.loads(Path(path).read_text())
    t, X, tc = _truth(d)
    if tc is None:
        return variant, None
    w, r = WicketSpec(), d["ball_radius_m"]
    xc = tc[1]
    true_hit = math.hypot(max(abs(xc[0]) - w.half_width_m, 0.0), max(xc[2] - w.top_m, 0.0) + max(-xc[2], 0.0)) <= r
    imp = np.where(X[:, 1] <= 1.5)[0]
    if not len(imp) or imp[0] == 0:
        return variant, None
    j = imp[0]
    a = (X[j - 1, 1] - 1.5) / (X[j - 1, 1] - X[j, 1])
    t_imp = float(t[j - 1] + a * (t[j] - t[j - 1]))
    rng = np.random.default_rng(zlib.crc32(d["clip_id"].encode()))
    pre = X[:, 1] > 1.5

    def n_visible(c):
        p = c.project(X[pre])
        return int(((p[:, 0] >= 0) & (p[:, 0] < c.image_width) & (p[:, 1] >= 0) & (p[:, 1] < c.image_height)).sum())

    side = max((manifest_camera(r) for r in side_rows), key=n_visible)
    views = []
    # Both cameras expose at the front camera's instants (hardware-synchronised, 60 fps), so no
    # cross-camera interpolation enters: this isolates the flight model.
    for cam, step, shift in ((camera_from_clip(d), 1, 0.0), (side, 1, sync_err_ms / 1000.0)):
        cam = cam.model_copy(update={"camera_id": cam.camera_id.split("@")[0]})
        idx = [k for k in range(0, len(t), step) if X[k, 1] > 1.5]
        uv, tt = [], []
        for k in idx:
            # the camera really exposes `shift` s later than its timestamp says: sample the truth there
            tk = t[k] + shift
            Xk = np.array([np.interp(tk, t, X[:, i]) for i in range(3)])
            p = cam.project(Xk[None])[0]
            if 0 <= p[0] < cam.image_width and 0 <= p[1] < cam.image_height:
                uv.append(tuple(p + rng.normal(0, 1.0, 2)))
                tt.append(t[k] * 1000)
        views.append(ViewObservations(camera=cam, t_ms=tt, uv=uv, sigma_px=1.0, hardware_sync=True))
    kw = dict(MV_VARIANTS[variant])
    if inflation is not None:
        kw["covariance_inflation"] = inflation
    mv = MultiViewConfig(**kw)
    try:
        res, diag, _ = analyze_lbw_multiview(views, LbwFacts(), LbwConfig(n_samples=500), mv, impact_time_ms=t_imp * 1000)
    except Exception as exc:  # report, never hide
        return variant, {"error": str(exc), "true_hit": true_hit}
    proj = res.wicket_projection
    err = None
    if proj and proj.nominal_crossing_xz:
        err = float(math.dist(proj.nominal_crossing_xz, [xc[0], xc[2]]))
    p05 = p95 = None
    if proj and proj.crossing:
        p05, p95 = proj.crossing.p05, proj.crossing.p95
    return variant, {"clip_id": d["clip_id"], "verdict": res.verdict.value, "true_hit": bool(true_hit), "err_m": err,
                     "p_hit": res.p_hit, "true_crossing_xz": [float(xc[0]), float(xc[2])],
                     "pred_crossing_xz": proj.nominal_crossing_xz if proj else None, "interval_p05": p05, "interval_p95": p95,
                     "n_bounces": d["delivery"].get("n_bounces"), "bounce_fitted": "bounce" in " ".join(res.reasons + res.warnings)}


def run_multiview(files, man, workers, sync_err_ms=0.0, inflation=None):
    side = [r for r in man.values() if r["camera"] == "side_on"]
    jobs = [(str(f), side, v, sync_err_ms, inflation) for v in MV_VARIANTS for f in files]
    res: dict[str, list] = {v: [] for v in MV_VARIANTS}
    with ProcessPoolExecutor(workers) as ex:
        for v, r in ex.map(_mv_one, jobs, chunksize=4):
            if r is not None:
                res[v].append(r)
    out = {}
    for v, rs in res.items():
        ok = [r for r in rs if "verdict" in r]
        definite = [r for r in ok if r["verdict"] in ("PROJECTED_TO_HIT", "PROJECTED_TO_MISS")]
        errs = np.array([r["err_m"] for r in ok if r["err_m"] is not None])
        cov = [lo[0] <= r["true_crossing_xz"][0] <= hi[0] and lo[1] <= r["true_crossing_xz"][1] <= hi[1]
               for r in ok if (lo := r.get("interval_p05")) and (hi := r.get("interval_p95"))]
        out[v] = {"n": len(rs), "errors": len(rs) - len(ok), "definite": len(definite),
                  "correct": sum(1 for r in definite if (r["verdict"] == "PROJECTED_TO_HIT") == r["true_hit"]),
                  "false_hit": sum(1 for r in definite if r["verdict"] == "PROJECTED_TO_HIT" and not r["true_hit"]),
                  "false_miss": sum(1 for r in definite if r["verdict"] == "PROJECTED_TO_MISS" and r["true_hit"]),
                  "true_hits": sum(1 for r in ok if r["true_hit"]),
                  "true_hits_confirmed": sum(1 for r in ok if r["true_hit"] and r["verdict"] == "PROJECTED_TO_HIT"),
                  "inconclusive_rate": (1 - len(definite) / len(ok)) if ok else None,
                  "stump_plane_error_median_m": float(np.median(errs)) if len(errs) else None,
                  "stump_plane_error_p90_m": float(np.percentile(errs, 90)) if len(errs) else None,
                  "interval_90_coverage": float(np.mean(cov)) if cov else None,
                  "covariance_inflation": inflation,
                  "confident_errors": [r for r in definite if (r["verdict"] == "PROJECTED_TO_HIT") != r["true_hit"]]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=240)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mv-inflation", type=float, default=None, help="multi-view covariance inflation (default: model default)")
    ap.add_argument("--skip", default="", help="comma list of experiments to skip: monocular,points3d,multiview")
    ap.add_argument("--out", default=str(ROOT / "outputs/reports/physics_refinement_synth.json"))
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    files, man = _front_files(a.offset, a.limit), _manifests()
    t0 = time.perf_counter()
    report = {"dataset": "Ranvin36/cricket-synth (synthetic, MIT)", "rig": "front_on_stump (+ manifest side_on for multiview)",
              "clips": f"front_on_stump clips {a.offset}..{a.offset + len(files) - 1} (held out from all tuning)",
              "caveat": "Synthetic deliveries and exact cameras: says which model terms help, not real-world accuracy."}
    if "points3d" not in skip:
        report["points3d_1cm"] = run_points3d(files)
        print(json.dumps(report["points3d_1cm"], indent=1), flush=True)
    if "multiview" not in skip:
        report["multiview_hw_sync"] = run_multiview(files, man, a.workers, inflation=a.mv_inflation)
        print(json.dumps(report["multiview_hw_sync"], indent=1), flush=True)
    if "monocular" not in skip:
        report["monocular"] = run_monocular(files, man, a.workers)
        print(json.dumps({k: {kk: v[kk] for kk in ("definite_verdicts", "false_hit", "false_miss", "true_hits_confirmed",
                                                   "true_hits", "inconclusive_rate", "stump_plane_error_m",
                                                   "interval_90_empirical_coverage", "drag_fit")}
                          for k, v in report["monocular"].items()}, indent=1, default=str), flush=True)
    report["runtime_s"] = round(time.perf_counter() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
