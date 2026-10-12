import numpy as np
import pytest

from ml.evaluation.synthetic import broadcast_camera, make_delivery
from ml.trajectory.ballistic import InsufficientObservations, fit_to_pixels, fit_to_points
from ml.trajectory.image_trajectory import fit_image_trajectory, predict


def test_quadratic_recovered():
    t = np.linspace(0, 0.6, 31)
    x, y = 100 + 400 * t, 300 - 200 * t + 250 * t**2
    fit = fit_image_trajectory(t, x, y, allow_change_point=False)
    assert fit.status == "ok"
    assert fit.model_name == "poly2"
    pr = predict(fit, np.array([0.3]))
    assert pr["x"][0] == pytest.approx(220, abs=0.01)
    assert pr["y"][0] == pytest.approx(262.5, abs=0.01)


def test_change_point_detected_for_bounce():
    t = np.linspace(0, 0.6, 31)
    tb = 0.3
    y = np.where(t < tb, 100 + 900 * t, 100 + 900 * tb - 500 * (t - tb))
    x = 50 + 300 * t
    fit = fit_image_trajectory(t, x, y)
    assert "changepoint" in fit.model_name
    assert fit.change_point_time_s == pytest.approx(tb, abs=0.03)


def test_robust_to_outlier_and_reports_it():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 0.6, 31)
    x = 100 + 400 * t + rng.normal(0, 0.5, t.size)
    y = 300 - 200 * t + 250 * t**2 + rng.normal(0, 0.5, t.size)
    y[10] += 80
    fit = fit_image_trajectory(t, x, y, allow_change_point=False)
    assert fit.inlier_mask[10] is False
    assert fit.residual_rms_px < 2.0


def test_insufficient_points():
    fit = fit_image_trajectory([0, 0.1, 0.2], [0, 1, 2], [0, 1, 2])
    assert fit.status == "insufficient_data"


def test_prediction_interval_grows_when_extrapolating():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 0.4, 21)
    x = 100 + 400 * t + rng.normal(0, 1, t.size)
    y = 300 + 100 * t + rng.normal(0, 1, t.size)
    fit = fit_image_trajectory(t, x, y, allow_change_point=False)
    pr = predict(fit, np.array([0.2, 0.8]))
    assert pr["sx"][1] > pr["sx"][0]


def test_ballistic_fit_to_3d_points_recovers_bounce():
    d = make_delivery(pitch_y=5.0, fps=200)
    rng = np.random.default_rng(0)
    X = d.xyz + rng.normal(0, 0.005, d.xyz.shape)
    fit = fit_to_points(d.times, X, ball_radius=d.model.ball_radius)
    assert fit.model.bounce
    true_tb, est_tb = d.model.bounce_time(), fit.model.bounce_time()
    assert est_tb == pytest.approx(true_tb, abs=0.005)
    assert np.allclose(fit.model.position(true_tb), d.model.position(true_tb), atol=0.03)
    assert fit.status == "ok"


def test_monocular_fit_reprojects_and_reports_uncertainty():
    """Single camera behind the bowler: pre-bounce flight and bounce are observable through the
    gravity + ground-contact constraints; post-bounce speed along the viewing direction is not.
    The fit must recover the former and *report* large uncertainty for the latter."""
    d = make_delivery(pitch_y=5.0, fps=100)
    rng = np.random.default_rng(0)
    uv = d.uv + rng.normal(0, 0.7, d.uv.shape)
    # The generating model has no drag or lift, so fit with matching physics.
    fit = fit_to_pixels(d.camera, d.times, uv, sigma_px=1.0, ball_radius=d.model.ball_radius, drag_c=0.0,
                        lift_prior_sigma=None, swing_prior_sigma=None)
    assert fit.mode == "monocular_approximate"
    assert fit.residual_rms < 1.5
    assert any("Monocular" in r for r in fit.reasons)
    tb = d.model.bounce_time()
    assert fit.model.bounce_time() == pytest.approx(tb, abs=0.01)
    assert np.linalg.norm(fit.model.position(tb)[0] - d.model.position(tb)[0]) < 0.3
    sigma = np.sqrt(np.diag(fit.cov))
    err = np.abs(fit.model.theta - d.model.theta)
    assert np.all(err <= 4 * sigma + 0.05)  # truth is inside the reported uncertainty
    assert sigma[7] > 10 * sigma[4]  # post-bounce along-pitch speed far less certain than pre-bounce


def test_monocular_needs_enough_points():
    with pytest.raises(InsufficientObservations):
        fit_to_pixels(broadcast_camera(), np.arange(4) * 0.02, np.zeros((4, 2)) + 600)


def test_monocular_radius_cue_and_lift_are_reported():
    d = make_delivery(pitch_y=5.0, fps=100)
    depth = (d.camera.R_np @ d.xyz.T).T[:, 2] + d.camera.t_np[2]
    radius = d.camera.K_np[0, 0] * d.model.ball_radius / depth
    fit = fit_to_pixels(d.camera, d.times, d.uv, sigma_px=1.0, ball_radius=d.model.ball_radius, radius_px=radius,
                        radius_sigma_px=0.5)
    assert len(fit.model.theta) == 11  # p0, v0, v1, a_swing, a_lift
    assert any("radius" in r for r in fit.reasons)
    assert any("lift" in r for r in fit.reasons) and any("swing" in r for r in fit.reasons)
    assert abs(fit.model.a_lift) < 3.0 and abs(fit.model.a_swing) < 2.0


def test_unobservable_parameters_are_uncertain_not_certain():
    """No observations after the bounce: post-bounce velocity must have huge, not zero, variance."""
    d = make_delivery(pitch_y=5.0, fps=200)
    tb = d.model.bounce_time()
    keep = d.times < tb - 0.01
    fit = fit_to_points(d.times[keep], d.xyz[keep], ball_radius=d.model.ball_radius, allow_bounce=False)
    assert not fit.model.bounce
    from ml.trajectory.ballistic import FlightFit, FlightModel

    m = FlightModel(np.concatenate([fit.model.theta, [0, -30, 3]]), fit.model.t0, True, d.model.ball_radius)
    J = np.zeros((6, 9))
    J[:, :6] = np.eye(6)
    from ml.trajectory.ballistic import WEAK_PRIOR_SIGMA

    cov = np.linalg.inv(J.T @ J + np.eye(9) / WEAK_PRIOR_SIGMA**2)
    ff = FlightFit(m, cov, "ok", "multiview_3d", 0.0, "m", 1.0, 6, 0.0, 0.1)
    assert np.sqrt(np.diag(ff.cov))[6:].min() > 10.0
