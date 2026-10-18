"""Experimental LBW analysis engine: combines geometry, uncertainty and rules.

Verdict semantics (geometric "would hit" question only):
* PROJECTED_TO_HIT  - P(ball volume intersects wicket) >= decision_confidence
* PROJECTED_TO_MISS - P(hit) <= 1 - decision_confidence
* INCONCLUSIVE      - uncertainty straddles the wicket boundary, or the data
                      quality rules out a definite answer
* INSUFFICIENT_DATA - no usable trajectory / projection

Rule checks are reported separately. Nothing here is an umpire's decision.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

from ml.calibration.pitch_geometry import BallSpec, WicketSpec
from ml.lbw.geometry import PointEstimate, WicketProjection, pitching_points, points_at, project_to_wicket
from ml.lbw.rules import LbwFacts, RuleCheck, RulesConfig, evaluate_rules
from ml.trajectory.ballistic import FlightFit

ENGINE_VERSION = "lbw-engine-0.1.0-experimental"
DISCLAIMER = (
    "Experimental research output. Not an official umpiring decision or a certified DRS result. "
    "Real-world accuracy of this system has not been established."
)


class LbwVerdict(StrEnum):
    PROJECTED_TO_HIT = "PROJECTED_TO_HIT"
    PROJECTED_TO_MISS = "PROJECTED_TO_MISS"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class LbwConfig(BaseModel):
    decision_confidence: float = 0.95
    n_samples: int = 2000
    require_validated_calibration_for_definite: bool = True
    max_projection_distance_m: float = 3.5
    max_flight_residual_px: float = 4.0
    # Model-uncertainty probe: re-evaluate with drag scaled by this factor (Cd 0.4 -> 0.6), or with the
    # default aerodynamic drag when the primary model was fitted without drag.
    alternative_drag_scale: float = 1.5
    # Multiplies the fitted parameter covariance before Monte Carlo sampling, to account for model
    # discrepancy (spin, seam, drag variation) that the measurement covariance does not capture.
    # Calibrated against empirical interval coverage on the cricket-synth benchmark (docs/EVALUATION.md).
    covariance_inflation: float = 2.0
    rules: RulesConfig = Field(default_factory=RulesConfig)


class LbwResult(BaseModel):
    verdict: LbwVerdict
    p_hit: float | None
    rules_summary: str
    rule_checks: list[RuleCheck]
    pitching_point: PointEstimate | None
    impact_point: PointEstimate | None
    impact_time_s: float | None
    impact_time_source: str | None
    wicket_projection: WicketProjection | None
    # Centre-of-ball / any-part zone probabilities for pitching and impact, plus impact distances. Consumed by
    # the DRS protocol layer (ml.protocol) for umpire's-call style tests; None when not computable.
    zone_probabilities: dict | None = None
    predicted_path: list[list[float]] = Field(default_factory=list, description="[t, x, y, z] from impact to stumps")
    observed_path_fit: list[list[float]] = Field(default_factory=list, description="[t, x, y, z] over observed span")
    measurement_uncertainty: dict = Field(default_factory=dict)
    model_uncertainty: dict = Field(default_factory=dict)
    input_quality: dict = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reconstruction_mode: str | None = None
    calibration_validated: bool = False
    engine_version: str = ENGINE_VERSION
    disclaimer: str = DISCLAIMER


def pitch_y_samples(models, impact_time_s: float) -> np.ndarray:
    ys = []
    for m in models:
        t = m.bounce_time()
        ys.append(m.position(t)[0][1] if t is not None and t <= impact_time_s else np.nan)
    return np.asarray(ys)


def _frac(mask: np.ndarray, n: int) -> float:
    return float(mask.sum() / n)


def _summary(v: np.ndarray) -> dict | None:
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "n_valid": int(len(v))}


def zone_probabilities(pitch_x: np.ndarray | None, impact_x: np.ndarray, impact_y: np.ndarray, pitch_y: np.ndarray | None,
                       half_width: float, radius: float, handed: str | None) -> dict:
    """Zone fractions over Monte Carlo samples (denominator = all samples; non-finite samples count in none).

    pitching: by ball centre (in line / outside off / outside leg).
    impact: centre in line; any part in line (centre within half_width + radius); no part in line, split by side.
    Off/leg splits need batter handedness and are None otherwise.
    """
    off = None if handed is None else (1.0 if handed == "right" else -1.0)
    out: dict = {"half_width_m": half_width, "ball_radius_m": radius, "handedness": handed}
    if pitch_x is not None and np.isfinite(pitch_x).any():
        n = len(pitch_x)
        fin = np.isfinite(pitch_x)
        out["pitching"] = {
            "centre_in_line": _frac(fin & (np.abs(np.nan_to_num(pitch_x)) <= half_width), n),
            "centre_outside_off": None if off is None else _frac(fin & (off * np.nan_to_num(pitch_x) > half_width), n),
            "centre_outside_leg": None if off is None else _frac(fin & (off * np.nan_to_num(pitch_x) < -half_width), n),
            "n_valid": int(fin.sum()), "n_samples": n,
        }
    else:
        out["pitching"] = None
    n = len(impact_x)
    fin = np.isfinite(impact_x)
    ax = np.abs(np.nan_to_num(impact_x, nan=1e9))
    sx = np.nan_to_num(impact_x, nan=0.0)
    out["impact"] = {
        "centre_in_line": _frac(fin & (ax <= half_width), n),
        "any_part_in_line": _frac(fin & (ax <= half_width + radius), n),
        "umpires_call_band_off": None if off is None else _frac(fin & (off * sx > half_width) & (off * sx <= half_width + radius), n),
        "umpires_call_band_leg": None if off is None else _frac(fin & (off * sx < -half_width) & (off * sx >= -(half_width + radius)), n),
        "no_part_outside_off": None if off is None else _frac(fin & (off * sx > half_width + radius), n),
        "no_part_outside_leg": None if off is None else _frac(fin & (off * sx < -(half_width + radius)), n),
        "n_valid": int(fin.sum()), "n_samples": n,
    }
    out["impact_distance_from_stumps_m"] = _summary(np.asarray(impact_y, dtype=float))
    if pitch_y is not None:
        out["pitch_to_impact_m"] = _summary(np.asarray(pitch_y, dtype=float) - np.asarray(impact_y, dtype=float))
    else:
        out["pitch_to_impact_m"] = None
    return out


def analyze_lbw(
    fit: FlightFit | None,
    facts: LbwFacts,
    config: LbwConfig | None = None,
    impact_time_s: float | None = None,
    calibration_validated: bool = False,
    wicket: WicketSpec = WicketSpec(),
    ball: BallSpec = BallSpec(),
) -> LbwResult:
    cfg = config or LbwConfig()
    cfg.rules.decision_confidence = cfg.decision_confidence
    reasons: list[str] = []
    warnings: list[str] = []

    def insufficient(msg: str) -> LbwResult:
        checks, summary = evaluate_rules(cfg.rules, facts, None, None, None, None, wicket.half_width_m, ball.radius_m)
        return LbwResult(verdict=LbwVerdict.INSUFFICIENT_DATA, p_hit=None, rules_summary=summary, rule_checks=checks,
                         pitching_point=None, impact_point=None, impact_time_s=impact_time_s, impact_time_source=None,
                         wicket_projection=None, reasons=[msg], warnings=warnings,
                         reconstruction_mode=fit.mode if fit else None, calibration_validated=calibration_validated)

    if fit is None:
        return insufficient("No ball-flight model available")
    if fit.status == "insufficient_data":
        return insufficient("Trajectory fit reported insufficient data")

    if cfg.covariance_inflation != 1.0:
        from dataclasses import replace

        fit = replace(fit, cov=fit.cov * cfg.covariance_inflation)

    if impact_time_s is None:
        impact_time_s = fit.t_last
        impact_src = "last_tracked_observation"
        warnings.append("Impact time not supplied: assumed to be the last tracked observation")
    else:
        impact_src = "user_supplied"
        if impact_time_s < fit.t_first:
            return insufficient("Impact time precedes the observed trajectory")
        if impact_time_s > fit.t_last + 0.05:
            warnings.append("Impact time is beyond the observed trajectory: the impact point itself is extrapolated")

    models = fit.sample_models(cfg.n_samples, np.random.default_rng(1))
    tb = fit.model.bounce_time()
    pitched = fit.model.bounce and tb is not None and tb <= impact_time_s
    pitch_est = pitching_points(models) if pitched else None
    pitch_x = None
    if pitched:
        pts = []
        for m in models:
            t = m.bounce_time()
            pts.append(m.position(t)[0][0] if t is not None and t <= impact_time_s else np.nan)
        pitch_x = np.asarray(pts)
    impact_est = points_at(models, impact_time_s)
    impact_x = np.asarray([m.position(impact_time_s)[0][0] for m in models])

    proj = project_to_wicket(fit, impact_time_s, wicket, cfg.n_samples)
    impact_y = np.asarray([m.position(impact_time_s)[0][1] for m in models])
    zone_p = zone_probabilities(pitch_x, impact_x, impact_y, pitch_y_samples(models, impact_time_s) if pitched else None,
                                wicket.half_width_m, ball.radius_m, facts.batter_handedness)
    p_hit = proj.p_hit

    # Model-uncertainty probe: same observations interpreted with an alternative drag assumption.
    m0 = fit.model
    model_unc: dict = {
        "primary_model": f"ballistic(drag_k={m0.drag_k}, drag_c={m0.drag_c:.4f}, lift={f'fitted {m0.a_lift:.2f}' if m0.lift else 'none'})",
        "alternatives_bic": fit.alternatives,
    }
    if cfg.alternative_drag_scale and proj.nominal_crossing_xz is not None:
        from dataclasses import replace

        from ml.lbw.geometry import time_at_plane_y
        from ml.trajectory.ballistic import DEFAULT_DRAG_C

        alt_c = m0.drag_c * cfg.alternative_drag_scale if m0.drag_c else DEFAULT_DRAG_C
        alt = replace(m0, drag_c=alt_c)
        tc = time_at_plane_y(alt, 0.0, impact_time_s)
        if tc is not None:
            p = alt.position(tc)[0]
            shift = float(np.hypot(p[0] - proj.nominal_crossing_xz[0], p[2] - proj.nominal_crossing_xz[1]))
            model_unc["stump_plane_shift_with_alt_drag_m"] = shift
            model_unc["note"] = f"Shift of the nominal stump-plane crossing if drag coefficient c={alt_c:.4f}/m is assumed"
            if shift > ball.radius_m:
                warnings.append(f"Stump-plane crossing moves {shift * 100:.1f} cm under an alternative drag model")

    input_quality = {
        "num_observations": fit.num_observations,
        "fit_status": fit.status,
        "residual_rms": fit.residual_rms,
        "residual_unit": fit.residual_unit,
        "condition_number": fit.condition_number,
        "observed_span_s": fit.t_last - fit.t_first,
    }
    meas_unc = {
        "crossing_std_m": proj.crossing.std if proj.crossing else None,
        "crossing_90pct_interval_m": [proj.crossing.p05, proj.crossing.p95] if proj.crossing else None,
        "impact_std_m": impact_est.std if impact_est else None,
        "n_samples": cfg.n_samples,
    }

    # ----- verdict
    if p_hit is None:
        verdict = LbwVerdict.INSUFFICIENT_DATA
        reasons.append("Projected path never reaches the stump plane in any sample")
    else:
        conf = cfg.decision_confidence
        if p_hit >= conf:
            verdict = LbwVerdict.PROJECTED_TO_HIT
        elif p_hit <= 1 - conf:
            verdict = LbwVerdict.PROJECTED_TO_MISS
        else:
            verdict = LbwVerdict.INCONCLUSIVE
            reasons.append(f"P(hit) = {p_hit:.3f}: uncertainty overlaps the wicket boundary")
        downgrade = []
        if fit.status != "ok":
            downgrade.append(f"trajectory fit status is '{fit.status}'")
        if cfg.require_validated_calibration_for_definite and not calibration_validated:
            downgrade.append("camera calibration is not validated")
        if fit.mode == "monocular_approximate":
            warnings.append("Single-camera reconstruction: depth comes from a physics assumption, not measurement")
        if proj.projection_distance_m and proj.projection_distance_m > cfg.max_projection_distance_m:
            downgrade.append(f"projection distance {proj.projection_distance_m:.2f} m exceeds {cfg.max_projection_distance_m} m")
        if proj.p_below_ground > 0.05:
            downgrade.append("projected path contacts the ground again before the stumps (second bounce not modelled)")
        if verdict in (LbwVerdict.PROJECTED_TO_HIT, LbwVerdict.PROJECTED_TO_MISS) and downgrade:
            reasons.append(f"Geometric result would be {verdict.value} (P(hit) = {p_hit:.3f}) but downgraded because " + "; ".join(downgrade))
            verdict = LbwVerdict.INCONCLUSIVE
        elif verdict != LbwVerdict.INCONCLUSIVE:
            reasons.append(f"P(hit) = {p_hit:.3f} with {cfg.n_samples} Monte Carlo samples")
        else:
            reasons.extend(f"Additional limitation: {d}" for d in downgrade)

    checks, summary = evaluate_rules(cfg.rules, facts, pitched, pitch_x, impact_x,
                                     p_hit if verdict in (LbwVerdict.PROJECTED_TO_HIT, LbwVerdict.PROJECTED_TO_MISS) else None,
                                     wicket.half_width_m, ball.radius_m)

    predicted, observed = [], []
    if proj.crossing_time_s is not None:
        t_end = proj.crossing_time_s.mean[0]
        for t in np.linspace(impact_time_s, t_end, 30):
            predicted.append([float(t), *map(float, fit.model.position(t)[0])])
    for t in np.linspace(fit.t_first, min(impact_time_s, fit.t_last), 40):
        observed.append([float(t), *map(float, fit.model.position(t)[0])])

    return LbwResult(
        verdict=verdict, p_hit=p_hit, rules_summary=summary, rule_checks=checks,
        pitching_point=pitch_est, impact_point=impact_est, impact_time_s=impact_time_s, impact_time_source=impact_src,
        wicket_projection=proj, zone_probabilities=zone_p, predicted_path=predicted, observed_path_fit=observed, measurement_uncertainty=meas_unc,
        model_uncertainty=model_unc, input_quality=input_quality, reasons=reasons + fit.reasons, warnings=warnings,
        reconstruction_mode=fit.mode, calibration_validated=calibration_validated,
    )
