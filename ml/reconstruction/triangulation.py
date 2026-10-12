"""Multi-view triangulation with association, quality checks and uncertainty."""

from __future__ import annotations

import itertools
import math

import numpy as np
from pydantic import BaseModel, Field

from ml.calibration.camera import CameraModel


class CameraObservation(BaseModel):
    camera_id: str
    frame_index: int
    timestamp_ms: float  # camera-local clock
    u: float
    v: float
    sigma_px: float = 2.0


class Point3D(BaseModel):
    timestamp_ms: float  # common clock
    xyz: tuple[float, float, float]
    cov: list[list[float]]
    reprojection_errors_px: dict[str, float]
    max_ray_angle_deg: float
    cameras: list[str]
    rejected_cameras: list[str] = Field(default_factory=list)
    sync_spread_ms: float
    quality: str  # "good" | "weak_geometry" | "high_error"


class ReconstructionResult(BaseModel):
    points: list[Point3D]
    unmatched_observations: int
    rejected_groups: int
    coordinate_frame: str = "pitch"
    calibration_validated: bool
    warnings: list[str] = Field(default_factory=list)


def triangulate_dlt(Ps: list[np.ndarray], uvs: list[np.ndarray]) -> np.ndarray:
    """Linear DLT triangulation (undistorted pixel coordinates)."""
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    # Row normalisation improves conditioning.
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        raise ValueError("Degenerate triangulation (point at infinity)")
    return X[:3] / X[3]


def refine_gauss_newton(Ps, uvs, X, iters: int = 10) -> np.ndarray:
    X = X.copy()
    for _ in range(iters):
        J, r = [], []
        for P, uv in zip(Ps, uvs):
            Xh = np.append(X, 1.0)
            p = P @ Xh
            proj = p[:2] / p[2]
            r.extend(proj - uv)
            dp = P[:, :3]
            J.extend([(dp[0] * p[2] - p[0] * dp[2]) / p[2] ** 2, (dp[1] * p[2] - p[1] * dp[2]) / p[2] ** 2])
        J, r = np.asarray(J), np.asarray(r)
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        X += step
        if np.linalg.norm(step) < 1e-9:
            break
    return X


def _jacobian(Ps, X) -> np.ndarray:
    J = []
    for P in Ps:
        p = P @ np.append(X, 1.0)
        dp = P[:, :3]
        J.extend([(dp[0] * p[2] - p[0] * dp[2]) / p[2] ** 2, (dp[1] * p[2] - p[1] * dp[2]) / p[2] ** 2])
    return np.asarray(J)


def triangulate_views(cams: list[CameraModel], obs: list[CameraObservation]) -> tuple[np.ndarray, np.ndarray, dict[str, float], float]:
    Ps = [c.P for c in cams]
    uvs = [c.undistort_points(np.array([o.u, o.v]))[0] for c, o in zip(cams, obs)]
    X = refine_gauss_newton(Ps, uvs, triangulate_dlt(Ps, uvs))
    J = _jacobian(Ps, X)
    W = np.diag(np.repeat([1.0 / o.sigma_px**2 for o in obs], 2))
    cov = np.linalg.pinv(J.T @ W @ J)
    errs = {}
    for c, _o, uv in zip(cams, obs, uvs):
        p = c.P @ np.append(X, 1.0)
        errs[c.camera_id] = float(np.linalg.norm(p[:2] / p[2] - uv))
    angle = 0.0
    for a, b in itertools.combinations(cams, 2):
        da, db = X - a.center_world, X - b.center_world
        cosang = np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db))
        angle = max(angle, math.degrees(math.acos(np.clip(cosang, -1, 1))))
    return X, cov, errs, angle


def associate_and_triangulate(
    cameras: list[CameraModel],
    observations: list[CameraObservation],
    sync_tolerance_ms: float = 2.0,
    max_reprojection_px: float = 5.0,
    min_ray_angle_deg: float = 5.0,
) -> ReconstructionResult:
    """Group observations by common-clock time, triangulate, and reject inconsistent views.

    Each camera's ``time_offset_ms`` maps its timestamps onto the common clock.
    Observations from different cameras are associated when their corrected
    timestamps differ by at most ``sync_tolerance_ms``. With 3+ cameras, the view
    with the worst reprojection error is dropped iteratively while it exceeds the
    threshold (simple consensus).
    """
    cams = {c.camera_id: c for c in cameras}
    warnings: list[str] = []
    unknown = {o.camera_id for o in observations} - set(cams)
    if unknown:
        raise ValueError(f"Observations reference unknown camera(s): {sorted(unknown)}")
    if len(cams) < 2:
        raise ValueError("Triangulation requires at least two calibrated cameras")
    if any(c.sync_uncertainty_ms is None for c in cameras):
        warnings.append("Synchronisation uncertainty not specified for every camera")
    for c in cameras:
        if c.sync_uncertainty_ms and c.sync_uncertainty_ms > sync_tolerance_ms:
            warnings.append(f"Camera {c.camera_id} sync uncertainty exceeds association tolerance")

    items = sorted(((o.timestamp_ms + cams[o.camera_id].time_offset_ms, o) for o in observations), key=lambda t: t[0])
    used = [False] * len(items)
    points: list[Point3D] = []
    rejected = 0
    for i, (t_i, o_i) in enumerate(items):
        if used[i]:
            continue
        group = {o_i.camera_id: (t_i, i)}
        for j in range(i + 1, len(items)):
            t_j, o_j = items[j]
            if t_j - t_i > sync_tolerance_ms:
                break
            if not used[j] and o_j.camera_id not in group:
                group[o_j.camera_id] = (t_j, j)
        if len(group) < 2:
            continue
        ids = list(group)
        dropped: list[str] = []
        while True:
            sel_cams = [cams[k] for k in ids]
            sel_obs = [items[group[k][1]][1] for k in ids]
            X, cov, errs, angle = triangulate_views(sel_cams, sel_obs)
            worst = max(errs, key=errs.get)
            if errs[worst] <= max_reprojection_px or len(ids) <= 2:
                break
            ids.remove(worst)
            dropped.append(worst)
        if errs[worst] > max_reprojection_px:
            rejected += 1
            continue
        if not all(cams[k].is_in_front(X)[0] for k in ids):
            rejected += 1
            continue
        for k in ids:
            used[group[k][1]] = True
        times = [group[k][0] for k in ids]
        quality = "good" if angle >= min_ray_angle_deg else "weak_geometry"
        points.append(Point3D(timestamp_ms=float(np.mean(times)), xyz=tuple(map(float, X)), cov=cov.tolist(),
                              reprojection_errors_px=errs, max_ray_angle_deg=angle, cameras=ids,
                              rejected_cameras=dropped, sync_spread_ms=float(np.ptp(times)), quality=quality))
    unmatched = sum(1 for u in used if not u)
    if not points:
        warnings.append("No observation groups could be triangulated")
    return ReconstructionResult(points=points, unmatched_observations=unmatched, rejected_groups=rejected,
                                calibration_validated=all(c.validated for c in cameras), warnings=warnings)
