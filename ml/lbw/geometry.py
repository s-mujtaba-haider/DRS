"""Pure geometry for LBW analysis (no cricket rules here).

Given a fitted flight model and its parameter uncertainty, compute by Monte
Carlo sampling:
* the pitching (bounce) point,
* the impact (interception) point at a given time,
* the projected crossing of the stump plane (y = 0) after impact, assuming the
  ball would have continued on its fitted path,
* the probability that the projected ball volume intersects the wicket.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from ml.calibration.pitch_geometry import WicketSpec
from ml.trajectory.ballistic import FlightFit, FlightModel


class PointEstimate(BaseModel):
    mean: list[float]
    std: list[float]
    p05: list[float]
    p95: list[float]
    n_valid: int


class WicketProjection(BaseModel):
    crossing: PointEstimate | None  # (x, z) at y = 0
    crossing_time_s: PointEstimate | None
    p_hit: float | None  # fraction of samples whose ball volume intersects the wicket
    p_centre_in_zone: float | None  # fraction with ball centre inside the wicket outline
    # P(centre inside the zone) for alternative zone heights used by different DRS protocol versions:
    # "top_of_bails" (= p_centre_in_zone) and "top_of_stumps". Protocol layer only (ml.protocol).
    p_centre_in_zone_by_height: dict[str, float] | None = None
    p_below_ground: float  # samples whose projection reached ground before the stumps
    n_samples: int
    projection_distance_m: float | None
    nominal_crossing_xz: list[float] | None


def _estimate(arr: np.ndarray) -> PointEstimate | None:
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if len(arr) == 0:
        return None
    return PointEstimate(mean=arr.mean(0).tolist(), std=arr.std(0).tolist(),
                         p05=np.percentile(arr, 5, axis=0).tolist(), p95=np.percentile(arr, 95, axis=0).tolist(),
                         n_valid=len(arr))


def time_at_plane_y(m: FlightModel, y_plane: float, t_from: float, t_max: float = 3.0, dt: float = 0.0005) -> float | None:
    ts = np.arange(t_from, t_from + t_max, dt)
    y = m.position(ts)[:, 1] - y_plane
    idx = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    if not len(idx):
        return None
    i = idx[0]
    a, b = ts[i], ts[i + 1]
    ya, yb = y[i], y[i + 1]
    return float(a + (b - a) * ya / (ya - yb))


def pitching_points(models: list[FlightModel]) -> PointEstimate | None:
    pts = []
    for m in models:
        tb = m.bounce_time()
        pts.append(m.position(tb)[0] if tb is not None else [np.nan] * 3)
    return _estimate(np.asarray(pts, dtype=float))


def points_at(models: list[FlightModel], t: float) -> PointEstimate | None:
    return _estimate(np.asarray([m.position(t)[0] for m in models]))


def project_to_wicket(fit: FlightFit, impact_time_s: float, wicket: WicketSpec, n_samples: int = 2000,
                      seed: int = 0) -> WicketProjection:
    models = fit.sample_models(n_samples, np.random.default_rng(seed))
    r = fit.model.ball_radius
    crossings, times = [], []
    below = 0
    for m in models:
        tc = time_at_plane_y(m, 0.0, impact_time_s)
        if tc is None:
            crossings.append([np.nan, np.nan])
            times.append([np.nan])
            continue
        pos = m.position(tc)[0]
        # Path below one radius means the ball would have bounced again: not modelled.
        ts = np.linspace(impact_time_s, tc, 50)
        if np.any(m.position(ts)[:, 2] < r - 1e-3):
            below += 1
        crossings.append([pos[0], pos[2]])
        times.append([tc])
    C = np.asarray(crossings)
    valid = np.all(np.isfinite(C), axis=1)
    if not valid.any():
        return WicketProjection(crossing=None, crossing_time_s=None, p_hit=None, p_centre_in_zone=None,
                                p_below_ground=below / n_samples, n_samples=n_samples, projection_distance_m=None,
                                nominal_crossing_xz=None)
    Cv = C[valid]
    hw, top = wicket.half_width_m, wicket.top_m
    # Ball volume intersects the wicket rectangle if its centre is within one radius of it.
    dx = np.maximum(np.abs(Cv[:, 0]) - hw, 0.0)
    dz = np.maximum(Cv[:, 1] - top, 0.0) + np.maximum(-Cv[:, 1], 0.0)
    hit = np.hypot(dx, dz) <= r
    # Reference only: ICC playing conditions (from June/July 2025) define the wicket zone as the
    # outline of stumps and bails; ball-tracking 'umpire's call' depends on whether the centre is inside it.
    centre_in = (np.abs(Cv[:, 0]) <= hw) & (Cv[:, 1] >= 0) & (Cv[:, 1] <= top)
    centre_in_stumps = (np.abs(Cv[:, 0]) <= hw) & (Cv[:, 1] >= 0) & (Cv[:, 1] <= wicket.stump_height_m)
    # Samples that never reach the stumps (e.g. ball moving away) count as misses.
    p_hit = float(hit.sum() / n_samples)
    nominal_tc = time_at_plane_y(fit.model, 0.0, impact_time_s)
    nominal = None
    dist = None
    if nominal_tc is not None:
        p = fit.model.position(nominal_tc)[0]
        nominal = [float(p[0]), float(p[2])]
        imp = fit.model.position(impact_time_s)[0]
        dist = float(math.dist(imp, p))
    return WicketProjection(crossing=_estimate(Cv), crossing_time_s=_estimate(np.asarray(times)[valid]),
                            p_hit=p_hit, p_centre_in_zone=float(centre_in.sum() / n_samples),
                            p_centre_in_zone_by_height={"top_of_bails": float(centre_in.sum() / n_samples),
                                                        "top_of_stumps": float(centre_in_stumps.sum() / n_samples)},
                            p_below_ground=below / n_samples, n_samples=n_samples, projection_distance_m=dist,
                            nominal_crossing_xz=nominal)
