"""Multi-camera LBW: per-camera 2D ball observations -> time alignment -> triangulation -> 3D flight fit -> LBW.

Cameras rarely expose at the same instants (different frame rates, free-running clocks), so observations
are aligned on the shared clock: every observation of the reference camera (the one with most points) is
paired with the other cameras' image positions *interpolated* between their two bracketing observed frames
(never across a gap longer than ``max_interp_gap_factor`` frame intervals). Interpolated image positions
are labelled as such in the diagnostics; they are not original observations.

Camera synchronisation error is a *systematic* time shift, not random noise, so it is propagated by
linearisation: the whole reconstruction and fit is repeated with each non-reference camera's offset moved
by its relative sync uncertainty, and the resulting parameter shifts are added to the fit covariance.
With software-only synchronisation a definite verdict is refused unless ball speed x sync uncertainty is
below ``max_sync_position_error_m`` (``configs/lbw.yaml``, section ``multiview``).
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

from ml.calibration.camera import CameraModel
from ml.calibration.pitch_geometry import BallSpec
from ml.lbw.engine import LbwConfig, LbwResult, analyze_lbw
from ml.lbw.rules import LbwFacts
from ml.reconstruction.triangulation import CameraObservation, ReconstructionResult, associate_and_triangulate
from ml.trajectory.ballistic import DEFAULT_DRAG_C, InsufficientObservations, fit_to_points


class ViewObservations(BaseModel):
    """Observed ball centres from one camera (camera-local timestamps)."""

    camera: CameraModel
    t_ms: list[float]
    uv: list[tuple[float, float]]
    frame_index: list[int] | None = None
    sigma_px: float = Field(2.0, gt=0)
    time_offset_ms: float = Field(0.0, description="local + offset = shared clock")
    sync_uncertainty_ms: float = Field(0.0, ge=0)
    hardware_sync: bool = False


class MultiViewConfig(BaseModel):
    max_interp_gap_factor: float = 2.5
    max_reprojection_px: float = 6.0
    min_points: int = 6
    point_sigma_floor_m: float = 0.005
    max_sync_position_error_m: float = 0.03
    outlier_sigma: float = 4.0
    use_drag: bool = True


class MultiViewDiagnostics(BaseModel):
    reference_camera: str
    cameras: list[dict]
    n_points: int
    n_points_used: int
    rejected_outliers: int
    interpolated_views: int
    reprojection_px_median: float | None
    ball_speed_mps: float | None
    relative_sync_uncertainty_ms: float
    sync_position_error_m: float | None
    hardware_sync: bool
    sync_definite_allowed: bool
    sync_covariance_added: bool
    points: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _median_interval(t: np.ndarray) -> float:
    d = np.diff(np.sort(t))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else math.inf


def _aligned_observations(views: list[ViewObservations], offsets: list[float], ref: int, mv: MultiViewConfig,
                          t_max_ms: float | None) -> tuple[list[CameraObservation], int]:
    """Shared-clock observations: reference camera as observed, other cameras interpolated to its instants."""
    rv = views[ref]
    T_ref = np.asarray(rv.t_ms) + offsets[ref]
    obs: list[CameraObservation] = []
    n_interp = 0
    others = []
    for k, v in enumerate(views):
        if k == ref:
            continue
        T = np.asarray(v.t_ms) + offsets[k]
        order = np.argsort(T)
        others.append((v, T[order], np.asarray(v.uv, float)[order], _median_interval(T)))
    for i, T in enumerate(T_ref):
        if t_max_ms is not None and T > t_max_ms + 0.5:
            continue
        group = [CameraObservation(camera_id=rv.camera.camera_id, frame_index=i, timestamp_ms=float(T),
                                   u=float(rv.uv[i][0]), v=float(rv.uv[i][1]), sigma_px=rv.sigma_px)]
        for v, Tw, UVw, dt in others:
            j = int(np.searchsorted(Tw, T))
            if j < len(Tw) and abs(Tw[j] - T) < 1e-3:
                uv, exact = UVw[j], True
            elif 0 < j < len(Tw) and Tw[j] - Tw[j - 1] <= mv.max_interp_gap_factor * dt:
                a = (T - Tw[j - 1]) / (Tw[j] - Tw[j - 1])
                uv, exact = (1 - a) * UVw[j - 1] + a * UVw[j], False
            else:
                continue
            n_interp += 0 if exact else 1
            group.append(CameraObservation(camera_id=v.camera.camera_id, frame_index=i, timestamp_ms=float(T),
                                           u=float(uv[0]), v=float(uv[1]), sigma_px=v.sigma_px))
        if len(group) >= 2:
            obs.extend(group)
    return obs, n_interp


def _reconstruct_and_fit(views, offsets, ref, mv, t_max_ms, force_bounce=None, t0=None):
    cams = [v.camera.model_copy(update={"time_offset_ms": 0.0}) for v in views]
    obs, n_interp = _aligned_observations(views, offsets, ref, mv, t_max_ms)
    if not obs:
        raise InsufficientObservations("No time-overlapping observations between the cameras")
    rec = associate_and_triangulate(cams, obs, sync_tolerance_ms=0.01, max_reprojection_px=mv.max_reprojection_px)
    pts = rec.points
    if len(pts) < mv.min_points:
        raise InsufficientObservations(f"{len(pts)} triangulated points; at least {mv.min_points} required")
    t = np.array([p.timestamp_ms / 1000.0 for p in pts])
    X = np.array([p.xyz for p in pts])
    sig = np.array([max(math.sqrt(max(np.trace(np.array(p.cov)) / 3, 0.0)), mv.point_sigma_floor_m) for p in pts])
    drag_c = DEFAULT_DRAG_C if mv.use_drag else 0.0
    ball_r = BallSpec().radius_m
    fit = fit_to_points(t, X, sig, ball_radius=ball_r, drag_c=drag_c, min_points=mv.min_points, force_bounce=force_bounce,
                        t0=t0)
    keep = np.ones(len(t), bool)
    r = np.linalg.norm(fit.model.position(t) - X, axis=1)
    bad = (r > mv.outlier_sigma * sig) & (r > 3 * np.median(r))
    if bad.any() and (~bad).sum() >= mv.min_points:
        keep = ~bad
        fit = fit_to_points(t[keep], X[keep], sig[keep], ball_radius=ball_r, drag_c=drag_c, min_points=mv.min_points,
                            force_bounce=fit.model.bounce if force_bounce is None else force_bounce, t0=fit.model.t0)
    return fit, rec, keep, n_interp


def analyze_lbw_multiview(views: list[ViewObservations], facts: LbwFacts, config: LbwConfig | None = None,
                          mv: MultiViewConfig | None = None, impact_time_ms: float | None = None
                          ) -> tuple[LbwResult, MultiViewDiagnostics | None, ReconstructionResult | None]:
    """LBW from two or more calibrated, time-aligned cameras. ``impact_time_ms`` is on the shared clock."""
    cfg = config or LbwConfig()
    mv = mv or MultiViewConfig()
    usable = [v for v in views if len(v.t_ms) >= 3 and len(v.t_ms) == len(v.uv)]
    if len(usable) < 2:
        res = analyze_lbw(None, facts, cfg)
        res.reasons.append("Multi-view analysis needs at least two cameras with three or more observations each")
        return res, None, None
    if len({v.camera.camera_id for v in usable}) != len(usable):
        raise ValueError("camera_id values must be unique")
    ref = int(np.argmax([len(v.t_ms) for v in usable]))
    offsets = [v.time_offset_ms for v in usable]
    try:
        fit, rec, keep, n_interp = _reconstruct_and_fit(usable, offsets, ref, mv, impact_time_ms)
    except InsufficientObservations as exc:
        res = analyze_lbw(None, facts, cfg)
        res.reasons.append(str(exc))
        res.reconstruction_mode = "multiview_3d"
        return res, None, None

    # ---- synchronisation uncertainty, propagated by re-fitting with shifted offsets
    warnings: list[str] = list(rec.warnings)
    u_ref = usable[ref].sync_uncertainty_ms
    rel = [math.hypot(u_ref, v.sync_uncertainty_ms) for k, v in enumerate(usable) if k != ref]
    u_rel = max(rel) if rel else 0.0
    sync_cov = np.zeros_like(fit.cov)
    added = False
    unpropagated: list[str] = []
    for k, v in enumerate(usable):
        if k == ref:
            continue
        u = math.hypot(u_ref, v.sync_uncertainty_ms)
        if u <= 0:
            continue
        shifted = list(offsets)
        shifted[k] += u
        try:
            f2, *_ = _reconstruct_and_fit(usable, shifted, ref, mv, impact_time_ms, force_bounce=fit.model.bounce,
                                          t0=fit.model.t0)
        except InsufficientObservations:
            unpropagated.append(v.camera.camera_id)
            continue
        if f2.model.theta.shape == fit.model.theta.shape and f2.model.t0 == fit.model.t0:
            d = f2.model.theta - fit.model.theta
            sync_cov += np.outer(d, d)
            added = True
        else:
            unpropagated.append(v.camera.camera_id)
    if added:
        from dataclasses import replace

        fit = replace(fit, cov=fit.cov + sync_cov)

    t_end = fit.t_last if impact_time_ms is None else impact_time_ms / 1000.0
    speed = float(np.linalg.norm(fit.model.velocity(np.array([min(t_end, fit.t_last)]))[0]))
    sync_err = speed * u_rel / 1000.0
    hw = all(v.hardware_sync for v in usable)
    extra: list[str] = []
    allowed = True
    if not hw:
        warnings.append(f"Software synchronisation: relative sync uncertainty {u_rel:.1f} ms x ball speed {speed:.1f} m/s "
                        f"= {sync_err * 100:.1f} cm possible position error")
        if sync_err > mv.max_sync_position_error_m:
            allowed = False
            extra.append(f"camera sync uncertainty ({u_rel:.1f} ms, about {sync_err * 100:.1f} cm at {speed:.1f} m/s) exceeds "
                         f"the {mv.max_sync_position_error_m * 100:.1f} cm tolerance for software-synchronised cameras")
    if unpropagated:
        allowed = False
        extra.append("the effect of synchronisation uncertainty could not be propagated for camera(s) "
                     + ", ".join(unpropagated))
    if impact_time_ms is None:
        warnings.append("Impact time not supplied: points after the pad impact may have been included in the fit")

    validated = all(v.camera.validated for v in usable)
    res = analyze_lbw(fit, facts, cfg, impact_time_s=None if impact_time_ms is None else impact_time_ms / 1000.0,
                      calibration_validated=validated, extra_downgrades=extra)
    res.warnings.extend(warnings)
    res.warnings.append(f"Multi-view reconstruction from {len(usable)} cameras; {n_interp} image position(s) of non-reference "
                        "cameras were interpolated between observed frames to the reference camera's instants")

    pts = rec.points
    reproj = [max(p.reprojection_errors_px.values()) for p in pts]
    diag = MultiViewDiagnostics(
        reference_camera=usable[ref].camera.camera_id,
        cameras=[{"camera_id": v.camera.camera_id, "n_observations": len(v.t_ms), "time_offset_ms": v.time_offset_ms,
                  "sync_uncertainty_ms": v.sync_uncertainty_ms, "hardware_sync": v.hardware_sync,
                  "calibration_validated": v.camera.validated} for v in usable],
        n_points=len(pts), n_points_used=int(keep.sum()), rejected_outliers=int((~keep).sum()), interpolated_views=n_interp,
        reprojection_px_median=float(np.median(reproj)) if reproj else None, ball_speed_mps=speed,
        relative_sync_uncertainty_ms=u_rel, sync_position_error_m=sync_err, hardware_sync=hw, sync_definite_allowed=allowed,
        sync_covariance_added=added,
        points=[{"t_ms": p.timestamp_ms, "xyz": list(p.xyz), "cov_trace_m2": float(np.trace(np.array(p.cov))),
                 "reprojection_px": p.reprojection_errors_px, "ray_angle_deg": p.max_ray_angle_deg, "used": bool(k)}
                for p, k in zip(pts, keep)],
        warnings=warnings,
    )
    return res, diag, rec
