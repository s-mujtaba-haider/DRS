"""2D (image-space) trajectory estimation.

Each image axis is modelled as a polynomial in time, optionally split at a
single change point (e.g. a bounce). Candidates (degree 1-3, with/without a
change point) are fit with Huber-weighted iteratively-reweighted least squares
and compared by BIC; leave-one-out error is reported as a validation signal.
Prediction intervals come from the parameter covariance plus residual noise.

Image-space fits describe how the ball moved *in the picture*. They are not a
metric 3D trajectory and are not used on their own for LBW decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats


class PolySegment(BaseModel):
    t_start_s: float
    t_end_s: float
    degree: int
    coef_x: list[float]  # highest power first (numpy.polyval order), t relative to t_ref
    coef_y: list[float]
    cov_x: list[list[float]]
    cov_y: list[list[float]]
    t_ref_s: float
    n_points: int


class ImageTrajectoryFit(BaseModel):
    status: str  # "ok" | "insufficient_data" | "poor_fit"
    model_name: str | None = None
    segments: list[PolySegment] = Field(default_factory=list)
    change_point_time_s: float | None = None
    residual_rms_px: float | None = None
    loo_rms_px: float | None = None
    bic: float | None = None
    candidates: list[dict] = Field(default_factory=list)
    inlier_mask: list[bool] = Field(default_factory=list)
    num_points: int = 0
    reasons: list[str] = Field(default_factory=list)


@dataclass
class _AxisFit:
    coef: np.ndarray
    cov: np.ndarray
    resid: np.ndarray
    weights: np.ndarray


def _huber_poly(t: np.ndarray, y: np.ndarray, deg: int, k: float = 1.5, iters: int = 20) -> _AxisFit:
    A = np.vander(t, deg + 1)
    w = np.ones_like(y)
    coef = np.zeros(deg + 1)
    for _ in range(iters):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = y - A @ coef
        mad = np.median(np.abs(r - np.median(r))) * 1.4826
        scale = max(mad, 1e-6)
        u = np.abs(r) / (k * scale)
        w_new = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        if np.allclose(w_new, w, atol=1e-4):
            w = w_new
            break
        w = w_new
    r = y - A @ coef
    dof = max(len(y) - (deg + 1), 1)
    sigma2 = float(np.sum(w * r**2) / dof)
    AtWA = (A * w[:, None]).T @ A
    cov = np.linalg.pinv(AtWA) * sigma2
    return _AxisFit(coef, cov, r, w)


def _fit_segment(t, x, y, deg, t_ref):
    tt = t - t_ref
    fx, fy = _huber_poly(tt, x, deg), _huber_poly(tt, y, deg)
    return fx, fy


def _bic(rss: float, n: int, k: int) -> float:
    rss = max(rss, 1e-9)
    return n * math.log(rss / n) + k * math.log(n)


def fit_image_trajectory(
    t_s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    degrees: tuple[int, ...] = (1, 2, 3),
    allow_change_point: bool = True,
    min_points: int = 5,
    min_segment_points: int = 4,
    max_rms_px: float = 6.0,
    outlier_sigma: float = 3.5,
) -> ImageTrajectoryFit:
    t_s, x, y = (np.asarray(a, dtype=float) for a in (t_s, x, y))
    order = np.argsort(t_s)
    t_s, x, y = t_s[order], x[order], y[order]
    n = len(t_s)
    if n < min_points:
        return ImageTrajectoryFit(status="insufficient_data", num_points=n,
                                  reasons=[f"Only {n} observations; at least {min_points} required"])

    candidates = []

    def evaluate(cp: int | None, deg: int):
        segs = [(0, n)] if cp is None else [(0, cp), (cp, n)]
        rss, k, fits = 0.0, 0, []
        for a, b in segs:
            if b - a < max(min_segment_points, deg + 2):
                return None
            t_ref = float(t_s[a])
            fx, fy = _fit_segment(t_s[a:b], x[a:b], y[a:b], deg, t_ref)
            rss += float(np.sum(fx.weights * fx.resid**2 + fy.weights * fy.resid**2))
            k += 2 * (deg + 1)
            fits.append((a, b, t_ref, fx, fy))
        if cp is not None:
            k += 1  # change-point location
        return _bic(rss, 2 * n, k), rss, fits

    configs: list[tuple[int | None, int]] = [(None, d) for d in degrees]
    if allow_change_point:
        configs += [(cp, d) for d in degrees for cp in range(min_segment_points, n - min_segment_points + 1)]
    best = None
    for cp, deg in configs:
        res = evaluate(cp, deg)
        if res is None:
            continue
        bic, rss, fits = res
        name = f"poly{deg}" + ("" if cp is None else "+changepoint")
        candidates.append({"model": name, "change_point_index": cp, "degree": deg, "bic": bic,
                           "rms_px": math.sqrt(rss / (2 * n))})
        if best is None or bic < best[0]:
            best = (bic, rss, fits, cp, deg, name)
    if best is None:
        return ImageTrajectoryFit(status="insufficient_data", num_points=n, reasons=["No candidate model could be fit"])

    bic, rss, fits, cp, deg, name = best
    # Outliers = points strongly down-weighted by the robust fit.
    inlier = np.ones(n, dtype=bool)
    for a, b, _, fx, fy in fits:
        scale = max(np.median(np.abs(np.concatenate([fx.resid, fy.resid]))) * 1.4826, 0.5)
        bad = (np.abs(fx.resid) > outlier_sigma * scale) | (np.abs(fy.resid) > outlier_sigma * scale)
        inlier[a:b] &= ~bad
    rms = math.sqrt(float(np.mean(np.concatenate([fx.resid[(inlier[a:b])] ** 2 for a, b, _, fx, _ in fits] +
                                                 [fy.resid[(inlier[a:b])] ** 2 for a, b, _, _, fy in fits]))))

    # Leave-one-out prediction error for the chosen structure (validation, not training error).
    loo_err = []
    for i in range(n):
        if not inlier[i]:
            continue
        mask = np.arange(n) != i
        seg_bounds = [(0, n)] if cp is None else [(0, cp), (cp, n)]
        for a, b in seg_bounds:
            if a <= i < b:
                sel = mask[a:b]
                if sel.sum() < deg + 2:
                    break
                t_ref = float(t_s[a])
                fx, fy = _fit_segment(t_s[a:b][sel], x[a:b][sel], y[a:b][sel], deg, t_ref)
                px = np.polyval(fx.coef, t_s[i] - t_ref)
                py = np.polyval(fy.coef, t_s[i] - t_ref)
                loo_err.append(math.hypot(px - x[i], py - y[i]))
    loo_rms = math.sqrt(float(np.mean(np.square(loo_err)))) if loo_err else None

    segments = [
        PolySegment(t_start_s=float(t_s[a]), t_end_s=float(t_s[b - 1]), degree=deg, coef_x=fx.coef.tolist(),
                    coef_y=fy.coef.tolist(), cov_x=fx.cov.tolist(), cov_y=fy.cov.tolist(), t_ref_s=t_ref,
                    n_points=b - a)
        for a, b, t_ref, fx, fy in fits
    ]
    reasons = [f"Selected {name} by BIC among {len(candidates)} candidates"]
    status = "ok"
    if rms > max_rms_px:
        status = "poor_fit"
        reasons.append(f"Residual RMS {rms:.2f}px exceeds {max_rms_px}px: observations do not support a smooth model")
    if (~inlier).sum():
        reasons.append(f"{int((~inlier).sum())} observation(s) flagged as outliers")
    return ImageTrajectoryFit(
        status=status, model_name=name, segments=segments,
        change_point_time_s=float(t_s[cp]) if cp is not None else None,
        residual_rms_px=rms, loo_rms_px=loo_rms, bic=bic,
        candidates=sorted(candidates, key=lambda c: c["bic"])[:10],
        inlier_mask=inlier.tolist(), num_points=n, reasons=reasons,
    )


def predict(fit: ImageTrajectoryFit, t_s: np.ndarray, level: float = 0.95) -> dict[str, np.ndarray]:
    """Evaluate the fit with prediction intervals. Times past the last segment extrapolate it."""
    if not fit.segments:
        raise ValueError("Fit has no segments")
    t_s = np.atleast_1d(np.asarray(t_s, dtype=float))
    out = {k: np.zeros_like(t_s) for k in ("x", "y", "sx", "sy")}
    z = stats.norm.ppf(0.5 + level / 2)
    noise = (fit.residual_rms_px or 0.0) ** 2
    for i, t in enumerate(t_s):
        seg = fit.segments[-1]
        for s in fit.segments:
            if t <= s.t_end_s:
                seg = s
                break
        v = np.vander([t - seg.t_ref_s], seg.degree + 1)[0]
        out["x"][i] = v @ np.asarray(seg.coef_x)
        out["y"][i] = v @ np.asarray(seg.coef_y)
        out["sx"][i] = math.sqrt(v @ np.asarray(seg.cov_x) @ v + noise)
        out["sy"][i] = math.sqrt(v @ np.asarray(seg.cov_y) @ v + noise)
    out["x_lo"], out["x_hi"] = out["x"] - z * out["sx"], out["x"] + z * out["sx"]
    out["y_lo"], out["y_hi"] = out["y"] - z * out["sy"], out["y"] + z * out["sy"]
    return out
