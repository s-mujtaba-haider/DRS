"""No-ball (Law 21.5) on synthetic geometry: front foot vs popping crease, back foot vs return crease."""

import numpy as np
import pytest

from ml.calibration.camera import look_at_camera
from ml.calibration.pitch_geometry import PitchSpec
from ml.noball import FootMark, NoBallFacts, analyze_no_ball
from ml.runout.crease import crease_from_camera, crease_from_points

P = PitchSpec()
L = P.stumps_distance_m
YC = L - P.popping_crease_from_stumps_m  # bowler's-end popping crease


def cam():
    # side-on camera square of the bowler's popping crease
    return look_at_camera("side", np.array([14.0, YC, 1.4]), np.array([0.0, YC, 0.0]), 2200.0, 1920, 1080)


def px(c, x, y):
    return tuple(c.project(np.array([[x, y, 0.0]]))[0])


def mark(c, y, frame=10, x=0.3):
    return FootMark(frame_index=frame, timestamp_ms=frame * 10.0, point=px(c, x, y), sigma_px=1.0)


def test_front_foot_behind_is_fair_and_beyond_is_no_ball():
    c = cam()
    line = crease_from_camera(c, "bowler")
    fair = analyze_no_ball(line, [mark(c, YC + 0.03)], facts=NoBallFacts(landing_frame_is_first_contact=True))
    assert fair.status == "FAIR" and fair.front_foot.signed_m == pytest.approx(0.03, abs=1e-3)
    nb = analyze_no_ball(line, [mark(c, YC - 0.02)])
    assert nb.status == "NO_BALL" and any("21.5.2" in r for r in nb.reasons)


def test_on_the_line_is_not_behind_and_near_line_is_inconclusive():
    c = cam()
    line = crease_from_camera(c, "bowler")
    on_line = FootMark(frame_index=3, timestamp_ms=30, point=px(c, 0.3, YC - 0.004), sigma_px=0.3)
    assert analyze_no_ball(line, [on_line]).status == "NO_BALL"  # on the white marking, precisely localised
    amb = analyze_no_ball(line, [FootMark(frame_index=3, timestamp_ms=30, point=px(c, 0.3, YC + 0.001), sigma_px=4.0)])
    assert amb.status == "INCONCLUSIVE"


def test_earliest_mark_is_landing_and_back_foot_rule():
    c = cam()
    line = crease_from_camera(c, "bowler")
    marks = [mark(c, YC - 0.05, frame=12), mark(c, YC + 0.04, frame=10)]  # foot slid forward after landing
    assert analyze_no_ball(line, marks).status == "FAIR"
    # Return crease (x = 1.32 m, along the pitch) seen from behind the bowler, where its lateral offset is visible.
    # Inside = towards the middle stump.
    back_cam = look_at_camera("behind", np.array([0.0, L + 18.0, 4.0]), np.array([0.0, YC, 0.0]), 2200.0, 1920, 1080)
    rx = P.return_crease_from_middle_m
    ret = crease_from_points(px(back_cam, rx, YC + 0.3), px(back_cam, rx, YC + 2.0), px(back_cam, 0.0, YC + 1.0))
    inside = FootMark(frame_index=8, timestamp_ms=80, point=px(back_cam, rx - 0.10, YC + 1.2), sigma_px=1.0)
    touching = FootMark(frame_index=8, timestamp_ms=80, point=px(back_cam, rx + 0.02, YC + 1.2), sigma_px=1.0)
    assert analyze_no_ball(line, [mark(c, YC + 0.05)], ret, [inside]).status == "FAIR"
    r = analyze_no_ball(line, [mark(c, YC + 0.05)], ret, [touching])
    assert r.status == "NO_BALL" and any("21.5.1" in x for x in r.reasons)


def test_missing_inputs_and_side_fact():
    c = cam()
    line = crease_from_camera(c, "bowler")
    assert analyze_no_ball(line, []).status == "INSUFFICIENT_DATA"
    wrong_side = analyze_no_ball(line, [mark(c, YC + 0.1)], facts=NoBallFacts(front_foot_correct_side_of_middle_stump_line=False))
    assert wrong_side.status == "NO_BALL"
