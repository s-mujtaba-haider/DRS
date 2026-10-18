"""Multi-camera LBW (synthetic, exact cameras): time alignment, triangulation, sync-uncertainty handling."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ml.calibration.camera import look_at_camera
from ml.evaluation.synthetic import make_delivery
from ml.lbw.engine import LbwConfig, LbwVerdict
from ml.lbw.multiview import MultiViewConfig, ViewObservations, analyze_lbw_multiview
from ml.lbw.rules import LbwFacts
from ml.trajectory.ballistic import DEFAULT_DRAG_C, fit_to_points

CFG = LbwConfig(n_samples=400)
FACTS = LbwFacts(batter_handedness="right")


def rig():
    a = look_at_camera("end_on", np.array([0.0, 42.0, 7.0]), np.array([0.0, 2.0, 0.4]), 4200.0, 1920, 1080)
    b = look_at_camera("side_on", np.array([32.0, 9.0, 4.0]), np.array([0.0, 8.0, 0.5]), 2600.0, 1920, 1080)
    for c in (a, b):
        c.validated = True
    return a, b


def truth(end_x: float):
    d = make_delivery(pitch_y=5.0, line_x=0.0, end_x=end_x, speed_mps=34.0, fps=1000)
    m = replace(d.model, drag_c=DEFAULT_DRAG_C)
    ts = np.arange(0, 1.5, 1e-4)
    return m, float(ts[np.argmax(m.position(ts)[:, 1] <= 1.2)])


def views(model, impact_t, err_ms=0.0, declared_ms=0.0, hw=True, b_local_shift_ms=0.0, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rig()
    ta = np.arange(0.0, impact_t, 1 / 50)
    tb = np.arange(0.003, impact_t, 1 / 100)
    uva = a.project(model.position(ta)) + rng.normal(0, noise, (len(ta), 2))
    uvb = b.project(model.position(tb + err_ms / 1000)) + rng.normal(0, noise, (len(tb), 2))
    return [
        ViewObservations(camera=a, t_ms=(ta * 1000).tolist(), uv=uva.tolist(), sigma_px=1.0, hardware_sync=hw),
        ViewObservations(camera=b, t_ms=(tb * 1000 + b_local_shift_ms).tolist(), uv=uvb.tolist(), sigma_px=1.0,
                         time_offset_ms=-b_local_shift_ms, sync_uncertainty_ms=declared_ms, hardware_sync=hw),
    ]


def test_hardware_synced_hit_and_miss():
    m, it = truth(0.0)
    res, diag, rec = analyze_lbw_multiview(views(m, it), FACTS, CFG, impact_time_ms=it * 1000)
    assert res.reconstruction_mode == "multiview_3d"
    assert res.verdict == LbwVerdict.PROJECTED_TO_HIT, res.reasons
    assert diag.sync_definite_allowed and diag.n_points >= 10 and diag.interpolated_views > 0
    x, z = res.wicket_projection.nominal_crossing_xz
    assert abs(x) < 0.05
    m2, it2 = truth(0.45)
    res2, *_ = analyze_lbw_multiview(views(m2, it2), FACTS, CFG, impact_time_ms=it2 * 1000)
    assert res2.verdict == LbwVerdict.PROJECTED_TO_MISS, res2.reasons


def test_software_sync_refuses_definite_verdict():
    m, it = truth(0.0)
    res, diag, _ = analyze_lbw_multiview(views(m, it, err_ms=20, declared_ms=20, hw=False), FACTS, CFG, impact_time_ms=it * 1000)
    assert res.verdict == LbwVerdict.INCONCLUSIVE
    assert not diag.sync_definite_allowed and diag.sync_position_error_m > 0.1
    assert any("sync uncertainty" in r for r in res.reasons)
    assert any("Software synchronisation" in w for w in res.warnings)


def test_sync_uncertainty_widens_projection():
    m, it = truth(0.0)
    loose = MultiViewConfig(max_sync_position_error_m=1.0)
    r0, *_ = analyze_lbw_multiview(views(m, it, err_ms=0, declared_ms=0, hw=False), FACTS, CFG, loose, it * 1000)
    r5, d5, _ = analyze_lbw_multiview(views(m, it, err_ms=0, declared_ms=5, hw=False), FACTS, CFG, loose, it * 1000)
    assert d5.sync_covariance_added
    assert np.linalg.norm(r5.wicket_projection.crossing.std) > 1.5 * np.linalg.norm(r0.wicket_projection.crossing.std)


def test_time_offsets_align_local_clocks():
    m, it = truth(0.0)
    r0, *_ = analyze_lbw_multiview(views(m, it), FACTS, CFG, impact_time_ms=it * 1000)
    r1, *_ = analyze_lbw_multiview(views(m, it, b_local_shift_ms=537.0), FACTS, CFG, impact_time_ms=it * 1000)
    assert r1.wicket_projection.nominal_crossing_xz == pytest.approx(r0.wicket_projection.nominal_crossing_xz, abs=1e-6)


def test_single_or_disjoint_views_are_insufficient():
    m, it = truth(0.0)
    v = views(m, it)
    res, diag, _ = analyze_lbw_multiview(v[:1], FACTS, CFG)
    assert res.verdict == LbwVerdict.INSUFFICIENT_DATA and diag is None
    v[1].time_offset_ms = 10_000.0  # no temporal overlap at all
    res, diag, _ = analyze_lbw_multiview(v, FACTS, CFG)
    assert res.verdict == LbwVerdict.INSUFFICIENT_DATA


def test_fit_to_points_drag_and_forced_bounce():
    m, it = truth(0.0)
    t = np.arange(0.0, it, 0.01)
    X = m.position(t)
    with_drag = fit_to_points(t, X, np.full(len(t), 0.005), drag_c=DEFAULT_DRAG_C)
    no_drag = fit_to_points(t, X, np.full(len(t), 0.005))
    assert with_drag.model.bounce and with_drag.residual_rms < no_drag.residual_rms
    assert with_drag.residual_rms < 0.005  # reported in metres even with per-point sigmas
    forced = fit_to_points(t, X, force_bounce=False)
    assert not forced.model.bounce


def test_bounce_time_cache_tracks_theta():
    m, _ = truth(0.0)
    tb = m.bounce_time()
    m.theta = m.theta.copy()
    m.theta[2] += 0.3  # higher release -> later bounce
    assert m.bounce_time() > tb
