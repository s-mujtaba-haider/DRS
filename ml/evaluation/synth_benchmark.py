"""Trajectory / stump-plane prediction benchmark on the cricket-synth dataset.

cricket-synth (Ranvin36/cricket-synth, MIT) provides physically simulated
deliveries (drag, Magnus, seam swing, bounce with friction) with exact 3D ball
positions, exact pixel projections and the camera model. The ball is not
intercepted, so the true stump-plane crossing is known.

Protocol per clip:
1. Convert the dataset frame (origin at pitch centre) into the DRS pitch frame
   (origin at striker's middle stump): y_drs = y_synth + L/2.
2. Choose a virtual impact where the ball reaches y = ``impact_y`` (a pad position).
3. Use only ground-truth pixel centres *before* that impact, plus optional
   Gaussian pixel noise (simulating detector localisation error).
4. Fit the monocular ballistic model, project from impact to y = 0, compare
   against the true crossing; compare the geometric verdict with the true hit/miss.

What this measures: the trajectory/reconstruction/LBW-projection layers under a
known camera with synthetic physics that is richer than the fitted model.
What it does NOT measure: detection or tracking on real footage, calibration
error on real footage, or real-world LBW accuracy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.calibration.camera import CameraModel
from ml.calibration.pitch_geometry import PitchSpec, WicketSpec
from ml.lbw.engine import LbwConfig, LbwVerdict, analyze_lbw
from ml.lbw.rules import LbwFacts
from ml.trajectory.ballistic import InsufficientObservations, fit_to_pixels

HALF_L = PitchSpec().stumps_distance_m / 2


@dataclass
class ClipResult:
    clip_id: str
    rig: str
    status: str
    n_obs: int = 0
    true_crossing_xz: list[float] | None = None
    pred_crossing_xz: list[float] | None = None
    crossing_error_m: float | None = None
    within_90pct_interval: bool | None = None
    true_hit: bool | None = None
    verdict: str | None = None
    p_hit: float | None = None
    projection_distance_m: float | None = None
    notes: list[str] = field(default_factory=list)


def load_clip(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def camera_from_clip(d: dict) -> CameraModel:
    c = d["camera"]
    intr, ext = c["intrinsics"], c["extrinsics_cv_frame1"]
    K = [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]]
    R = np.asarray(ext["R"], float)
    t = np.asarray(ext["t"], float)
    # x_cam = R X_s + t, X_s = X_d - [0, L/2, 0]  =>  t_d = t - R @ [0, L/2, 0]
    t_d = t - R @ np.array([0.0, HALF_L, 0.0])
    return CameraModel(camera_id=f"synth:{d['clip_id']}", image_width=intr["width"], image_height=intr["height"],
                       K=K, R=R.tolist(), t=t_d.tolist(), calibration_type="imported", validated=True,
                       assumptions=["Exact synthetic camera from dataset ground truth (frame-1 extrinsics)"])


def true_crossing(times: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray] | None:
    y = X[:, 1]
    idx = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    if not len(idx):
        return None
    i = idx[0]
    a = y[i] / (y[i] - y[i + 1])
    return float(times[i] + a * (times[i + 1] - times[i])), X[i] + a * (X[i + 1] - X[i])


def evaluate_clip(d: dict, impact_y: float = 1.5, pixel_noise: float = 1.0, seed: int = 0,
                  lbw_cfg: LbwConfig | None = None, use_radius: bool = True, radius_noise: float = 0.5,
                  use_prior: bool = True) -> ClipResult:
    rig = d["camera"]["rig"]
    res = ClipResult(d["clip_id"], rig, "ok")
    frames = [f for f in d["frames"] if f["ball_in_scene"] and f["world_m"] is not None]
    if not frames:
        res.status = "no_ball_frames"
        return res
    t = np.array([f["t_s"] for f in frames])
    X = np.array([f["world_m"] for f in frames], float)
    X[:, 1] += HALF_L
    tc = true_crossing(t, X)
    if tc is None:
        res.status = "ball_never_reaches_stumps"
        return res
    _, Xc = tc
    res.true_crossing_xz = [float(Xc[0]), float(Xc[2])]
    w = WicketSpec()
    r = d["ball_radius_m"]
    dx = max(abs(Xc[0]) - w.half_width_m, 0.0)
    dz = max(Xc[2] - w.top_m, 0.0) + max(-Xc[2], 0.0)
    res.true_hit = bool(math.hypot(dx, dz) <= r)

    before = [i for i, f in enumerate(frames) if X[i, 1] > impact_y and f["visible"] and f["centre_px"] is not None]
    after = [i for i in range(len(frames)) if X[i, 1] <= impact_y]
    if not after:
        res.status = "no_impact_point"
        return res
    i_imp = after[0]
    # Impact time: interpolate where y crosses impact_y.
    j = i_imp - 1
    a = (X[j, 1] - impact_y) / (X[j, 1] - X[i_imp, 1])
    t_imp = float(t[j] + a * (t[i_imp] - t[j]))
    res.n_obs = len(before)
    rng = np.random.default_rng(seed)
    uv = np.array([frames[i]["centre_px"] for i in before], float)
    uv = uv + rng.normal(0, pixel_noise, uv.shape) if pixel_noise > 0 else uv
    cam = camera_from_clip(d)
    radius = None
    if use_radius:
        radius = np.array([frames[i]["radius_px"] or np.nan for i in before], float)
        radius = radius + rng.normal(0, radius_noise, radius.shape) if radius_noise > 0 else radius
    try:
        fit = fit_to_pixels(cam, t[before], uv, sigma_px=max(pixel_noise, 0.5), ball_radius=r, radius_px=radius,
                            radius_sigma_px=max(radius_noise, 0.25),
                            bounce_retention_prior=(0.8, 0.15) if use_prior else None)
    except InsufficientObservations as exc:
        res.status = "insufficient_observations"
        res.notes.append(str(exc))
        return res
    out = analyze_lbw(fit, LbwFacts(), lbw_cfg or LbwConfig(n_samples=500), impact_time_s=t_imp,
                      calibration_validated=True)
    res.verdict = out.verdict.value
    res.p_hit = out.p_hit
    proj = out.wicket_projection
    if proj and proj.nominal_crossing_xz:
        res.pred_crossing_xz = proj.nominal_crossing_xz
        res.crossing_error_m = float(math.dist(proj.nominal_crossing_xz, res.true_crossing_xz))
        res.projection_distance_m = proj.projection_distance_m
        if proj.crossing:
            lo, hi = proj.crossing.p05, proj.crossing.p95
            res.within_90pct_interval = bool(lo[0] <= Xc[0] <= hi[0] and lo[1] <= Xc[2] <= hi[1])
    else:
        res.status = "no_projection"
    return res


def summarize(results: list[ClipResult]) -> dict:
    ok = [r for r in results if r.crossing_error_m is not None]
    errs = np.array([r.crossing_error_m for r in ok]) if ok else np.array([])
    cov = [r.within_90pct_interval for r in ok if r.within_90pct_interval is not None]
    verdicts = [r for r in results if r.verdict and r.true_hit is not None]
    cm = {"true_hit": {}, "true_miss": {}}
    for r in verdicts:
        key = "true_hit" if r.true_hit else "true_miss"
        cm[key][r.verdict] = cm[key].get(r.verdict, 0) + 1
    definite = [r for r in verdicts if r.verdict in (LbwVerdict.PROJECTED_TO_HIT.value, LbwVerdict.PROJECTED_TO_MISS.value)]
    correct = sum(1 for r in definite if (r.verdict == LbwVerdict.PROJECTED_TO_HIT.value) == r.true_hit)
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    return {
        "num_clips": len(results),
        "status_counts": statuses,
        "num_with_projection": len(ok),
        "stump_plane_error_m": None if not ok else {
            "median": float(np.median(errs)), "mean": float(errs.mean()), "p90": float(np.percentile(errs, 90)),
            "max": float(errs.max()),
        },
        "interval_90_empirical_coverage": float(np.mean(cov)) if cov else None,
        "verdict_confusion": cm,
        "inconclusive_rate": (sum(1 for r in verdicts if r.verdict == LbwVerdict.INCONCLUSIVE.value) / len(verdicts)) if verdicts else None,
        "definite_verdicts": len(definite),
        "definite_verdict_accuracy": (correct / len(definite)) if definite else None,
        "false_hit": sum(1 for r in definite if r.verdict == LbwVerdict.PROJECTED_TO_HIT.value and not r.true_hit),
        "false_miss": sum(1 for r in definite if r.verdict == LbwVerdict.PROJECTED_TO_MISS.value and r.true_hit),
        "by_rig": {rig: sum(1 for r in results if r.rig == rig) for rig in sorted({r.rig for r in results})},
    }
