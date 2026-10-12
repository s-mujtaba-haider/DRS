import numpy as np
import pytest

from ml.calibration.pitch_geometry import WicketSpec
from ml.evaluation.synthetic import make_delivery
from ml.lbw.engine import LbwConfig, LbwVerdict, analyze_lbw
from ml.lbw.rules import CheckStatus, LbwFacts, side_probabilities
from ml.trajectory.ballistic import FlightFit, fit_to_points


def _exact_fit(d, sigma=1e-4):
    """A FlightFit around the true model with tiny covariance (validated 3D case)."""
    k = len(d.model.theta)
    return FlightFit(d.model, np.eye(k) * sigma**2, "ok", "multiview_3d", 0.001, "m", 1.0, len(d.times),
                     float(d.times[0]), float(d.times[-1]))


FACTS = LbwFacts(batter_handedness="right", no_ball=False, ball_touched_bat_first=False,
                 intercepted_by_person_not_bat_hand=True, genuine_attempt_to_play=True)


def test_straight_ball_projected_to_hit():
    d = make_delivery(line_x=0.0, end_x=0.0, pitch_y=4.0)
    res = analyze_lbw(_exact_fit(d), FACTS, impact_time_s=d.impact_time, calibration_validated=True)
    assert res.verdict == LbwVerdict.PROJECTED_TO_HIT
    assert res.p_hit > 0.99
    assert res.rules_summary == "ALL_CONDITIONS_MET"
    x, z = res.wicket_projection.nominal_crossing_xz
    assert abs(x) < 0.02 and 0 < z < WicketSpec().top_m


def test_wide_ball_projected_to_miss():
    d = make_delivery(line_x=0.05, end_x=0.45, pitch_y=4.0)
    res = analyze_lbw(_exact_fit(d), FACTS, impact_time_s=d.impact_time, calibration_validated=True)
    assert res.verdict == LbwVerdict.PROJECTED_TO_MISS
    assert res.rules_summary == "CONDITIONS_NOT_MET"


def test_bouncing_over_is_a_miss():
    d = make_delivery(line_x=0.0, end_x=0.0, pitch_y=6.5, post_bounce_vz=6.0)
    res = analyze_lbw(_exact_fit(d), FACTS, impact_time_s=d.impact_time, calibration_validated=True)
    assert res.wicket_projection.nominal_crossing_xz[1] > WicketSpec().top_m
    assert res.verdict == LbwVerdict.PROJECTED_TO_MISS


def test_marginal_ball_with_uncertainty_is_inconclusive():
    hw = WicketSpec().half_width_m
    d = make_delivery(line_x=0.0, end_x=hw + 0.036, pitch_y=4.0)  # centre one radius outside: clipping the edge
    fit = _exact_fit(d, sigma=0.05)
    res = analyze_lbw(fit, FACTS, impact_time_s=d.impact_time, calibration_validated=True)
    assert res.verdict == LbwVerdict.INCONCLUSIVE
    assert 0.05 < res.p_hit < 0.95


def test_unvalidated_calibration_downgrades_definite_result():
    d = make_delivery(line_x=0.0, end_x=0.0, pitch_y=4.0)
    res = analyze_lbw(_exact_fit(d), FACTS, impact_time_s=d.impact_time, calibration_validated=False)
    assert res.verdict == LbwVerdict.INCONCLUSIVE
    assert any("not validated" in r for r in res.reasons)
    relaxed = analyze_lbw(_exact_fit(d), FACTS, LbwConfig(require_validated_calibration_for_definite=False),
                          impact_time_s=d.impact_time, calibration_validated=False)
    assert relaxed.verdict == LbwVerdict.PROJECTED_TO_HIT


def test_pitched_outside_leg_fails_rule_for_right_hander():
    d = make_delivery(line_x=-0.4, end_x=0.0, pitch_y=4.0)  # -x is leg side for a right-hander
    res = analyze_lbw(_exact_fit(d), FACTS, impact_time_s=d.impact_time, calibration_validated=True)
    pitching = next(c for c in res.rule_checks if c.id == "pitching")
    assert pitching.status == CheckStatus.FAIL
    # Same ball to a left-hander pitches outside off: allowed.
    lh = analyze_lbw(_exact_fit(d), FACTS.model_copy(update={"batter_handedness": "left"}),
                     impact_time_s=d.impact_time, calibration_validated=True)
    assert next(c for c in lh.rule_checks if c.id == "pitching").status == CheckStatus.PASS


def test_unknown_facts_stay_unknown():
    d = make_delivery(line_x=0.0, end_x=0.0, pitch_y=4.0)
    res = analyze_lbw(_exact_fit(d), LbwFacts(), impact_time_s=d.impact_time, calibration_validated=True)
    statuses = {c.id: c.status for c in res.rule_checks}
    assert statuses["not_bat_first"] == CheckStatus.UNKNOWN
    assert statuses["pitching"] == CheckStatus.UNKNOWN  # handedness unknown -> leg side undefined
    assert res.rules_summary == "UNDETERMINED"


def test_no_fit_is_insufficient_data():
    res = analyze_lbw(None, FACTS)
    assert res.verdict == LbwVerdict.INSUFFICIENT_DATA
    assert "Not an official" in res.disclaimer


def test_uncertainty_propagates_from_noisy_3d_fit():
    d = make_delivery(line_x=0.0, end_x=0.05, pitch_y=4.0, fps=100)
    rng = np.random.default_rng(3)
    small = fit_to_points(d.times, d.xyz + rng.normal(0, 0.003, d.xyz.shape), ball_radius=d.model.ball_radius)
    big = fit_to_points(d.times, d.xyz + rng.normal(0, 0.03, d.xyz.shape), ball_radius=d.model.ball_radius)
    r_small = analyze_lbw(small, FACTS, impact_time_s=d.impact_time, calibration_validated=True)
    r_big = analyze_lbw(big, FACTS, impact_time_s=d.impact_time, calibration_validated=True)
    assert r_big.wicket_projection.crossing.std[0] > r_small.wicket_projection.crossing.std[0]


def test_side_probabilities():
    x = np.array([0.0, 0.3, -0.3, np.nan])
    sp = side_probabilities(x, 0.1143, 0.036, "right", True)
    assert sp["in_line"] == pytest.approx(1 / 3)
    assert sp["outside_off"] == pytest.approx(1 / 3)
    assert sp["outside_leg"] == pytest.approx(1 / 3)
