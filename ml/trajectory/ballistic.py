"""Physics-informed 3D ball-flight model.

Model: piecewise ballistic flight under gravity with optional linear drag,
with at most one bounce. Pre-bounce state (p0, v0) is free; the bounce time is
where the ball centre descends to one ball-radius above the ground; the
post-bounce velocity v1 is free (captures seam/spin deviation and energy loss
without asserting a particular bounce law). Positions are continuous.

Two fitting modes:
* ``fit_to_points``: 3D observations (e.g. multi-view triangulation).
* ``fit_to_pixels``: a single calibrated camera. Depth is recovered only through
  the physical model (known gravity) and is therefore APPROXIMATE; the returned
  covariance usually shows weak depth observability.

Parameter covariance comes from the Gauss-Newton approximation and is sampled to
propagate uncertainty to any derived quantity (pitching point, stump plane, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ml.calibration.camera import CameraModel

G = 9.80665
WEAK_PRIOR_SIGMA = 100.0


# Quadratic-drag coefficient c = rho * Cd * A / (2 m) for a men's cricket ball
# (rho 1.225 kg/m^3, Cd ~0.4, radius 3.6 cm, mass 0.16 kg) -> ~0.0062 1/m.
# Deceleration ~ c * v^2 (about 9 m/s^2 at 140 km/h). Cd varies with seam, wear and speed.
DEFAULT_DRAG_C = 1.225 * 0.4 * np.pi * 0.036**2 / (2 * 0.16)


@dataclass
class FlightModel:
    """Piecewise flight model.

    theta = [p0(3), v0(3)] (+ [v1(3)] when bounce) (+ [a_swing] when swing) (+ [a_lift] when lift).
    Drag is approximated per segment as linear drag with k = drag_k + drag_c * |v_segment_start|.
    ``a_lift`` / ``a_swing`` are extra constant vertical / lateral (x) accelerations (Magnus, seam swing,
    drift), each fitted with a zero-mean prior.
    """

    theta: np.ndarray
    t0: float
    bounce: bool
    ball_radius: float
    drag_k: float = 0.0  # 1/s, constant linear drag
    drag_c: float = 0.0  # 1/m, quadratic-drag coefficient (linearised per segment)
    lift: bool = False
    swing: bool = False
    _tb_cache: tuple | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def _n_base(self) -> int:
        return 9 if self.bounce else 6

    @property
    def a_swing(self) -> float:
        return float(self.theta[self._n_base]) if self.swing else 0.0

    @property
    def a_lift(self) -> float:
        return float(self.theta[self._n_base + int(self.swing)]) if self.lift else 0.0

    def _k(self, v) -> float:
        return self.drag_k + self.drag_c * float(np.linalg.norm(v))

    def _free(self, p, v, tau):
        tau = np.asarray(tau, dtype=float)[..., None]
        g = np.array([self.a_swing, 0.0, -G + self.a_lift])
        k = self._k(v)
        if k <= 1e-9:
            return p + v * tau + 0.5 * g * tau**2, v + g * tau
        e = np.exp(-k * tau)
        pos = p + v * (1 - e) / k + g * (tau / k - (1 - e) / k**2)
        vel = v * e + g * (1 - e) / k
        return pos, vel

    def bounce_time(self) -> float | None:
        if not self.bounce:
            return None
        # Cached per parameter vector: position() needs it on every call and the root search dominates
        # Monte Carlo cost. The key covers every input, so in-place edits to theta cannot return stale values.
        key = (self.theta.tobytes(), self.t0, self.ball_radius, self.drag_k, self.drag_c)
        if self._tb_cache is not None and self._tb_cache[0] == key:
            return self._tb_cache[1]
        tb = self._solve_bounce_time()
        self._tb_cache = (key, tb)
        return tb

    def _solve_bounce_time(self) -> float | None:
        p0, v0 = self.theta[:3], self.theta[3:6]
        # Solve z(tau) = r numerically (handles drag); bracket scan then bisection.
        taus = np.linspace(1e-4, 3.0, 3001)
        z = self._free(p0, v0, taus)[0][:, 2] - self.ball_radius
        idx = np.where((z[:-1] > 0) & (z[1:] <= 0))[0]
        if not len(idx):
            return None
        a, b = taus[idx[0]], taus[idx[0] + 1]
        for _ in range(50):
            m = 0.5 * (a + b)
            if self._free(p0, v0, m)[0][2] - self.ball_radius > 0:
                a = m
            else:
                b = m
        return self.t0 + 0.5 * (a + b)

    def _segments(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        p0, v0 = self.theta[:3], self.theta[3:6]
        pos, vel = self._free(p0, v0, t - self.t0)
        tb = self.bounce_time()
        if tb is not None:
            pb, _ = self._free(p0, v0, tb - self.t0)
            after = t > tb
            if np.any(after):
                pa, va = self._free(pb, self.theta[6:9], t[after] - tb)
                pos[after], vel[after] = pa, va
        return pos, vel

    def position(self, t) -> np.ndarray:
        return self._segments(t)[0]

    def velocity(self, t) -> np.ndarray:
        return self._segments(t)[1]

    def with_theta(self, theta) -> FlightModel:
        return FlightModel(np.asarray(theta, float), self.t0, self.bounce, self.ball_radius, self.drag_k, self.drag_c, self.lift,
                           self.swing)


@dataclass
class FlightFit:
    model: FlightModel
    cov: np.ndarray
    status: str  # "ok" | "poor_fit" | "ill_conditioned" | "insufficient_data"
    mode: str  # "multiview_3d" | "monocular_approximate"
    residual_rms: float  # metres (3D) or pixels (monocular)
    residual_unit: str
    condition_number: float
    num_observations: int
    t_first: float
    t_last: float
    reasons: list[str] = field(default_factory=list)
    alternatives: dict[str, float] = field(default_factory=dict)  # model name -> BIC

    def sample_models(self, n: int, rng: np.random.Generator | None = None) -> list[FlightModel]:
        rng = rng or np.random.default_rng(0)
        cov = 0.5 * (self.cov + self.cov.T)
        w, V = np.linalg.eigh(cov)
        cov = (V * np.clip(w, 0, None)) @ V.T  # nearest PSD matrix (removes numerical noise)
        try:
            thetas = rng.multivariate_normal(self.model.theta, cov, size=n, method="eigh")
        except (np.linalg.LinAlgError, ValueError):
            thetas = np.repeat(self.model.theta[None], n, axis=0)
        return [self.model.with_theta(th) for th in thetas]

    def summary(self) -> dict:
        tb = self.model.bounce_time()
        return {
            "status": self.status, "mode": self.mode, "bounce": self.model.bounce,
            "bounce_time_s": tb, "residual_rms": self.residual_rms, "residual_unit": self.residual_unit,
            "condition_number": self.condition_number, "num_observations": self.num_observations,
            "drag_k": self.model.drag_k, "drag_c": self.model.drag_c, "a_lift": self.model.a_lift, "a_swing": self.model.a_swing, "reasons": self.reasons, "alternatives_bic": self.alternatives,
            "theta": self.model.theta.tolist(), "theta_sigma": np.sqrt(np.clip(np.diag(self.cov), 0, None)).tolist(),
        }


def _finish(res, model: FlightModel, n_resid: int, mode: str, unit: str, n_obs: int, t: np.ndarray, max_rms: float,
            noise_floor: float, resid_scale: float = 1.0) -> FlightFit:
    """``resid_scale`` converts (possibly whitened) residuals back to ``unit`` for reporting."""
    J = res.jac
    dof = max(n_resid - len(model.theta), 1)
    s2 = max(2 * res.cost / dof, noise_floor**2)
    JtJ = J.T @ J
    cond = float(np.linalg.cond(JtJ))
    # A pseudo-inverse would give *zero* variance along unobservable directions (falsely certain).
    # A weak prior (sigma = WEAK_PRIOR_SIGMA in parameter units, i.e. m, m/s, m/s^2) keeps such
    # directions very uncertain instead, which propagates into wide Monte Carlo spreads.
    cov = np.linalg.inv(JtJ + np.eye(len(JtJ)) / WEAK_PRIOR_SIGMA**2) * s2
    rms = math.sqrt(2 * res.cost / n_resid) * resid_scale
    reasons, status = [], "ok"
    if rms > max_rms:
        status = "poor_fit"
        reasons.append(f"Residual RMS {rms:.3g}{unit} exceeds {max_rms}{unit}")
    if cond > 1e12:
        status = "ill_conditioned" if status == "ok" else status
        reasons.append(f"Normal matrix condition number {cond:.2e}: parameters weakly observable")
    return FlightFit(model, cov, status, mode, rms, unit, cond, n_obs, float(t.min()), float(t.max()), reasons)


def _bic(cost: float, n: int, k: int) -> float:
    return n * math.log(max(2 * cost, 1e-12) / n) + k * math.log(n)


def fit_to_points(t: np.ndarray, X: np.ndarray, sigma_m: np.ndarray | None = None, ball_radius: float = 0.0364,
                  drag_k: float = 0.0, allow_bounce: bool = True, min_points: int = 6, max_rms_m: float = 0.05,
                  drag_c: float = 0.0, force_bounce: bool | None = None, t0: float | None = None) -> FlightFit:
    """Fit the flight model to 3D points (multi-view reconstruction).

    ``sigma_m`` (per point) whitens the residuals; the parameter covariance is then scaled by the reduced
    chi-square but never below the stated sigmas, and the reported RMS is converted back to metres.
    ``force_bounce`` fits only the bounce (True) or no-bounce (False) model instead of choosing by BIC.
    ``t0`` fixes the parameter time origin (default: first point), so fits of perturbed data are comparable.
    """
    t, X = np.asarray(t, float), np.asarray(X, float)
    n = len(t)
    if n < min_points:
        raise InsufficientObservations(f"{n} 3D points; at least {min_points} required")
    w = np.ones(n) if sigma_m is None else 1.0 / np.asarray(sigma_m, float)
    t0 = float(t.min()) if t0 is None else float(t0)
    # Initial guess from a quadratic fit to the first half.
    half = max(3, n // 2)
    A = np.vstack([np.ones(half), t[:half] - t0]).T
    sol, *_ = np.linalg.lstsq(A, X[:half], rcond=None)
    p0, v0 = sol[0], sol[1]
    fits = {}
    options = [force_bounce] if force_bounce is not None else ([False, True] if allow_bounce else [False])
    for bounce in options:
        theta0 = np.concatenate([p0, v0] + ([v0 * np.array([1, 1, -0.6])] if bounce else []))

        def resid(th, bounce=bounce):
            m = FlightModel(th, t0, bounce, ball_radius, drag_k, drag_c)
            if bounce and m.bounce_time() is None:
                return np.full(3 * n, 1e3)
            return ((m.position(t) - X) * w[:, None]).ravel()

        res = least_squares(resid, theta0, method="lm", x_scale="jac", max_nfev=4000)
        model = FlightModel(res.x, t0, bounce, ball_radius, drag_k, drag_c)
        if bounce and model.bounce_time() is None:
            continue
        fits["bounce" if bounce else "no_bounce"] = (res, model)
    if not fits:
        raise InsufficientObservations("No physically plausible flight model fits the 3D points")
    best_name = min(fits, key=lambda k: _bic(fits[k][0].cost, 3 * n, len(fits[k][1].theta)))
    res, model = fits[best_name]
    if sigma_m is None:
        out = _finish(res, model, 3 * n, "multiview_3d", "m", n, t, max_rms_m, 1e-4)
    else:  # whitened residuals: floor the reduced chi-square at 1 and report RMS in metres
        out = _finish(res, model, 3 * n, "multiview_3d", "m", n, t, max_rms_m, 1.0,
                      resid_scale=float(np.sqrt(np.mean(np.asarray(sigma_m, float) ** 2))))
    out.alternatives = {k: _bic(r.cost, 3 * n, len(m.theta)) for k, (r, m) in fits.items()}
    out.reasons.insert(0, f"Selected {best_name} model by BIC")
    return out


class InsufficientObservations(ValueError):
    pass


def _ray_plane_z(cam: CameraModel, uv, z: float) -> np.ndarray | None:
    o, d = cam.ray(uv)
    if abs(d[2]) < 1e-9:
        return None
    s = (z - o[2]) / d[2]
    return o + s * d if s > 0 else None


def _ray_at_depth(cam: CameraModel, uv, depth: float) -> np.ndarray:
    """World point on the pixel's ray whose camera-frame z equals ``depth``."""
    o, d = cam.ray(uv)
    dz = float((cam.R_np @ d)[2])
    return o + d * (depth / max(dz, 1e-6))


# Along-pitch speed retained through the bounce. Pitch-impact studies on cricket balls report
# substantial, surface-dependent horizontal speed loss; this broad prior (mean 0.8, sd 0.15)
# only regularises the otherwise weakly observable post-bounce depth in single-camera fits.
DEFAULT_BOUNCE_RETENTION = (0.8, 0.15)


def fit_to_pixels(cam: CameraModel, t: np.ndarray, uv: np.ndarray, sigma_px: float = 2.0, ball_radius: float = 0.0364,
                  drag_k: float = 0.0, allow_bounce: bool = True, min_points: int = 8,
                  max_rms_px: float = 4.0, radius_px: np.ndarray | None = None, radius_sigma_px: float = 0.75,
                  bounce_retention_prior: tuple[float, float] | None = DEFAULT_BOUNCE_RETENTION,
                  drag_c: float = DEFAULT_DRAG_C, lift_prior_sigma: float | None = 3.0,
                  swing_prior_sigma: float | None = 2.0) -> FlightFit:
    """Monocular, physics-constrained reconstruction (APPROXIMATE).

    Depth information comes from (a) gravity, (b) the ground-contact constraint at the
    bounce, (c) optionally the apparent ball radius in pixels (depth = f * r / r_px) and
    (d) optionally a weak prior on along-pitch speed retained through the bounce.
    Each source that is used is listed in the returned reasons.
    """
    t, uv = np.asarray(t, float), np.asarray(uv, float)
    n = len(t)
    if n < min_points:
        raise InsufficientObservations(f"{n} image observations; at least {min_points} required")
    t0 = float(t.min())
    span = float(t.max() - t0)
    if span <= 0:
        raise InsufficientObservations("Observations span zero time")
    f_px = float(cam.K_np[0, 0])
    rad = None
    if radius_px is not None:
        rad = np.asarray(radius_px, float)
        rad = np.where(np.isfinite(rad) & (rad > 0.5), rad, np.nan)
        if np.isfinite(rad).sum() < 3:
            rad = None
    depth_obs = None if rad is None else f_px * ball_radius / rad

    lift = lift_prior_sigma is not None
    swing = swing_prior_sigma is not None

    def resid_parts(th, bounce):
        m = FlightModel(th, t0, bounce, ball_radius, drag_k, drag_c, lift, swing)
        if bounce and m.bounce_time() is None:
            return None
        X = m.position(t)
        Xc = (cam.R_np @ X.T).T + cam.t_np
        if np.any(Xc[:, 2] <= 0.1):
            return None
        parts = [((cam.project(X) - uv) / sigma_px).ravel()]
        if rad is not None:
            r = (f_px * ball_radius / Xc[:, 2] - rad) / radius_sigma_px
            parts.append(np.where(np.isfinite(r), r, 0.0))
        if bounce and bounce_retention_prior is not None:
            tb = m.bounce_time()
            vb = m._free(th[:3], th[3:6], tb - t0)[1]
            mu, sd = bounce_retention_prior
            parts.append(np.array([(th[7] / vb[1] - mu) / sd]) if abs(vb[1]) > 1e-3 else np.zeros(1))
        if swing:
            parts.append(np.array([m.a_swing / swing_prior_sigma]))
        if lift:
            parts.append(np.array([m.a_lift / lift_prior_sigma]))
        return parts

    n_extra = n if rad is not None else 0
    fits = {}
    for bounce in ([False, True] if allow_bounce else [False]):
        n_res = 2 * n + n_extra + (1 if bounce and bounce_retention_prior else 0) + (1 if lift else 0) + (1 if swing else 0)

        def resid(th, bounce=bounce, n_res=n_res):
            parts = resid_parts(th, bounce)
            return np.full(n_res, 1e3) if parts is None else np.concatenate(parts)

        k_end = n - 1 if not bounce else max(2, int(0.6 * n))
        starts = []
        if depth_obs is not None and np.isfinite(depth_obs[0]) and np.isfinite(depth_obs[k_end]):
            starts.append((_ray_at_depth(cam, uv[0], depth_obs[0]), _ray_at_depth(cam, uv[k_end], depth_obs[k_end])))
        for z_start in (1.2, 2.0, 2.6):
            for z_end in (0.1, 0.5, 1.0):
                a, b = _ray_plane_z(cam, uv[0], z_start), _ray_plane_z(cam, uv[k_end], z_end)
                if a is not None and b is not None:
                    starts.append((a, b))
        best = None
        for a, b in starts:
            dt = max(t[k_end] - t0, 1e-3)
            v0 = (b - a) / dt
            v0[2] += 0.5 * G * dt  # undo gravity drop over the interval
            theta0 = np.concatenate([a, v0] + ([v0 * np.array([1, 0.8, -0.5])] if bounce else []) + ([[0.0]] if swing else []) + ([[0.0]] if lift else []))
            try:
                res = least_squares(resid, theta0, method="trf", x_scale="jac", max_nfev=3000)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or res.cost < best.cost:
                best = res
        if best is None:
            continue
        model = FlightModel(best.x, t0, bounce, ball_radius, drag_k, drag_c, lift, swing)
        if bounce and model.bounce_time() is None:
            continue
        fits["bounce" if bounce else "no_bounce"] = (best, model)
    if not fits:
        raise InsufficientObservations("No physically plausible flight model could be initialised from the rays")

    bics = {k: _bic(r.cost, 2 * n + n_extra, len(m.theta)) for k, (r, m) in fits.items()}
    best_name = min(bics, key=bics.get)
    res, model = fits[best_name]
    # Residuals are whitened: a reduced chi-square floor of 1 keeps the covariance from
    # shrinking below the assumed measurement noise.
    out = _finish(res, model, len(res.fun), "monocular_approximate", "px", n, t, max_rms_px, 1.0, resid_scale=sigma_px)
    out.residual_rms = math.sqrt(float(np.mean(res.fun[: 2 * n] ** 2))) * sigma_px  # pixel residuals only
    out.reasons = [r for r in out.reasons if not r.startswith("Residual RMS")]
    if out.residual_rms > max_rms_px:
        out.status = "poor_fit"
        out.reasons.append(f"Pixel residual RMS {out.residual_rms:.2f}px exceeds {max_rms_px}px")
    elif out.status == "poor_fit":
        out.status = "ok"
    out.alternatives = bics
    out.reasons.insert(0, f"Selected {best_name} model by BIC")
    out.reasons.append("Monocular reconstruction: depth inferred from physical constraints, not measured by a second view")
    if rad is not None:
        out.reasons.append(f"Apparent ball radius used as a depth cue (sigma {radius_sigma_px}px)")
    if best_name == "bounce" and bounce_retention_prior is not None:
        out.reasons.append("Assumption: along-pitch speed retention through bounce ~ N({:.2f}, {:.2f})".format(*bounce_retention_prior))
    if drag_c:
        out.reasons.append(f"Aerodynamic drag modelled with fixed coefficient c={drag_c:.4f}/m (Cd~0.4 assumed)")
    if swing:
        out.reasons.append(f"Lateral swing/drift acceleration fitted: {model.a_swing:.2f} m/s^2 (prior 0 +/- {swing_prior_sigma})")
    if lift:
        out.reasons.append(f"Vertical lift acceleration fitted: {model.a_lift:.2f} m/s^2 (prior 0 +/- {lift_prior_sigma})")
    if not cam.validated:
        out.reasons.append("Camera calibration is approximate/unvalidated")
    return out
