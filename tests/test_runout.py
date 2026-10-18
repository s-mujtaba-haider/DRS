"""Run-out / stumping: crease geometry, wicket-break detection, crease engine and Law 38/39 rules (synthetic)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml.calibration.camera import look_at_camera
from ml.calibration.pitch_geometry import PitchSpec
from ml.lbw.rules import CheckStatus
from ml.runout.crease import CreaseGeometryError, crease_from_camera, crease_from_points, measure_point
from ml.runout.engine import BatterObservation, CreaseFacts, CreaseStatus, WicketBreak, analyze_crease_event
from ml.runout.events import detect_wicket_break
from ml.runout.rules import DismissalFacts, analyze_runout, analyze_stumping

L = PitchSpec().stumps_distance_m


def side_cam(y=1.22):
    # square-leg style side-on camera at the striker's end
    return look_at_camera("side", np.array([12.0, y, 1.6]), np.array([0.0, y, 0.0]), 2000.0, 1920, 1080)


def px(cam, x, y, z=0.0):
    return tuple(cam.project(np.array([[x, y, z]]))[0])


# ------------------------------------------------------------------ crease geometry
def test_metric_crease_distance_and_sign_striker_end():
    cam = side_cam()
    line = crease_from_camera(cam, "striker")
    behind = measure_point(line, px(cam, 0.3, 1.00), sigma_px=1.0)
    assert behind.basis == "metric" and behind.state == "BEHIND"
    assert behind.signed_m == pytest.approx(0.22, abs=1e-3)
    short = measure_point(line, px(cam, 0.3, 1.40), sigma_px=1.0)
    assert short.state == "NOT_BEHIND" and short.signed_m == pytest.approx(-0.18, abs=1e-3)
    assert short.signed_px < 0 < behind.signed_px  # image and metric signs agree


def test_bowler_end_sign_convention():
    cam = side_cam(y=L - 1.22)
    line = crease_from_camera(cam, "bowler")
    m = measure_point(line, px(cam, 0.0, L - 1.0), sigma_px=1.0)
    assert m.state == "BEHIND" and m.signed_m == pytest.approx(0.22, abs=1e-3)


def test_near_line_point_is_ambiguous_not_guessed():
    cam = side_cam()
    line = crease_from_camera(cam, "striker")
    m = measure_point(line, px(cam, 0.0, 1.22 - 0.002), sigma_px=3.0)
    assert m.state == "AMBIGUOUS" and 0.05 < m.p_behind < 0.95
    assert m.sigma_m > 0.002


def test_point_on_crease_line_is_not_behind():
    # Law 7.3: the crease is the back edge; on the line is not behind it.
    cam = side_cam()
    line = crease_from_camera(cam, "striker")
    m = measure_point(line, px(cam, 0.0, PitchSpec().popping_crease_from_stumps_m), sigma_px=0.5)
    assert m.p_behind == pytest.approx(0.5, abs=0.02) and m.state == "AMBIGUOUS"


def test_manual_crease_pixels():
    line = crease_from_points((100, 500), (900, 520), behind_point=(500, 700))
    assert line.signed_px((500, 600)) > 0 > line.signed_px((500, 400))
    m = measure_point(line, (500, 560), sigma_px=2.0)
    assert m.basis == "image" and m.state == "BEHIND"
    with pytest.raises(CreaseGeometryError):
        crease_from_points((100, 500), (102, 500), (0, 0))
    with pytest.raises(CreaseGeometryError):
        crease_from_points((100, 500), (900, 500), (500, 500.5))


# --------------------------------------------------------------- wicket-break detection
def stumps_frame(bail_off: bool, rng, occlude: bool = False):
    img = np.full((240, 320, 3), (40, 110, 40), np.uint8)
    for x in (140, 158, 176):
        cv2.rectangle(img, (x, 80), (x + 6, 200), (235, 235, 235), -1)
    cv2.rectangle(img, (139, 74), (163, 78), (60, 200, 230), -1)
    if bail_off:
        cv2.rectangle(img, (200, 40), (224, 44), (60, 200, 230), -1)
    else:
        cv2.rectangle(img, (160, 74), (183, 78), (60, 200, 230), -1)
    if occlude:
        cv2.circle(img, (165, 120), 30, (30, 30, 200), -1)
    noise = rng.normal(0, 2.0, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def test_detects_bail_removal_frame():
    rng = np.random.default_rng(0)
    frames = [(i, i * 20.0, stumps_frame(i >= 10, rng)) for i in range(20)]
    det = detect_wicket_break(frames, (130, 66, 195, 205))
    assert det.status == "BREAK_CANDIDATE"
    assert (det.last_intact_frame, det.first_broken_frame) == (9, 10)
    assert det.interval_ms == pytest.approx(20.0)
    wb = WicketBreak.from_detection(det)
    assert wb.source == "auto" and not wb.confirmed


def test_detects_break_in_encoded_video(tmp_path):
    from ml.runout.events import detect_wicket_break_in_video
    from ml.video.ingest import write_video

    rng = np.random.default_rng(3)
    path = write_video((stumps_frame(i >= 12, rng) for i in range(24)), tmp_path / "stumps.mp4", 50)
    det = detect_wicket_break_in_video(path, (130, 66, 195, 205))
    assert det.status == "BREAK_CANDIDATE" and det.first_broken_frame == 12
    assert det.interval_ms == pytest.approx(20.0, abs=1.0)  # container timestamps at 50 fps


def test_transient_occlusion_not_reported():
    rng = np.random.default_rng(1)
    frames = [(i, i * 20.0, stumps_frame(False, rng, occlude=(i == 8))) for i in range(16)]
    det = detect_wicket_break(frames, (130, 66, 195, 205))
    assert det.status == "NO_CHANGE_DETECTED"
    assert any(f.flagged for f in det.frames)  # the occlusion was seen, but did not persist


def test_too_few_frames():
    rng = np.random.default_rng(2)
    det = detect_wicket_break([(i, i * 20.0, stumps_frame(False, rng)) for i in range(3)], (130, 66, 195, 205))
    assert det.status == "INSUFFICIENT_DATA"
    assert WicketBreak.from_detection(det) is None


# --------------------------------------------------------------------- crease engine
DT = 20.0  # 50 fps


def obs(cam, frame, y, grounded):
    return BatterObservation(frame_index=frame, timestamp_ms=frame * DT, point=px(cam, 0.2, y), point_sigma_px=1.0, grounded=grounded)


def wb(a=7, b=8):
    return WicketBreak(last_intact_frame=a, last_intact_ms=a * DT, first_broken_frame=b, first_broken_ms=b * DT, confirmed=True)


@pytest.fixture
def setup():
    cam = side_cam()
    return cam, crease_from_camera(cam, "striker")


def test_in_ground_before_break(setup):
    cam, line = setup
    o = [obs(cam, 3, 1.8, True), obs(cam, 4, 1.5, True), obs(cam, 5, 1.0, True), obs(cam, 6, 0.9, True),
         obs(cam, 7, 0.85, True), obs(cam, 8, 0.8, True)]
    r = analyze_crease_event(line, o, wb())
    assert r.status == CreaseStatus.IN_GROUND
    assert [e.kind for e in r.timeline] == ["grounded_behind_crease", "wicket_broken"]
    assert any("50 fps" in w for w in r.warnings)


def test_out_of_ground(setup):
    cam, line = setup
    o = [obs(cam, f, 2.6 - 0.15 * f, True) for f in range(3, 10)]  # y = 1.55 at frame 7, 1.40 at frame 8
    r = analyze_crease_event(line, o, wb())
    assert r.status == CreaseStatus.OUT_OF_GROUND
    assert "short of the crease" in r.reasons[0]


def test_grounding_in_break_window_is_inconclusive(setup):
    cam, line = setup
    o = [obs(cam, 6, 1.6, True), obs(cam, 7, 1.35, False), obs(cam, 8, 1.05, True)]
    r = analyze_crease_event(line, o, wb())
    assert r.status == CreaseStatus.INCONCLUSIVE
    assert r.grounding_window_ms == (140.0, 160.0)
    assert "overlap" in r.reasons[0]


def test_bouncing_bat_law_30_1_2(setup):
    cam, line = setup
    o = [obs(cam, 4, 1.4, True), obs(cam, 5, 1.0, True), obs(cam, 6, 0.9, False), obs(cam, 7, 0.8, False), obs(cam, 8, 0.7, False)]
    assert analyze_crease_event(line, o, wb()).status == CreaseStatus.INCONCLUSIVE
    r = analyze_crease_event(line, o, wb(), CreaseFacts(running_or_diving_into_ground=True))
    assert r.status == CreaseStatus.IN_GROUND and "30.1.2" in r.reasons[0]


def test_airborne_behind_at_break_is_inconclusive(setup):
    cam, line = setup
    o = [obs(cam, 7, 1.4, False), obs(cam, 8, 1.1, False)]
    assert analyze_crease_event(line, o, wb()).status == CreaseStatus.INCONCLUSIVE


def test_insufficient_data(setup):
    cam, line = setup
    assert analyze_crease_event(line, [obs(cam, 5, 1.0, True)], None).status == CreaseStatus.INSUFFICIENT_DATA
    blank = [BatterObservation(frame_index=5, timestamp_ms=100.0)]
    assert analyze_crease_event(line, blank, wb()).status == CreaseStatus.INSUFFICIENT_DATA


def test_wicket_break_order_validated():
    with pytest.raises(ValueError):
        WicketBreak(last_intact_frame=8, last_intact_ms=160, first_broken_frame=8, first_broken_ms=160)


# ---------------------------------------------------------------------- Law 38 / 39
def test_runout_all_conditions(setup):
    cam, line = setup
    o = [obs(cam, f, 2.6 - 0.15 * f, True) for f in range(3, 10)]
    facts = DismissalFacts(batter="non_striker", wicket_broken_fairly=True, ball_touched_fielder_before_break=True,
                           left_ground_to_avoid_injury=False)
    r = analyze_runout(line, o, wb(), facts)
    assert r.status == CreaseStatus.OUT_OF_GROUND and r.rules_summary == "ALL_CONDITIONS_MET"
    r2 = analyze_runout(line, o, wb(), DismissalFacts())
    assert r2.rules_summary == "UNDETERMINED"
    r3 = analyze_runout(line, o, wb(), facts.model_copy(update={"ball_touched_fielder_before_break": False}))
    assert r3.rules_summary == "CONDITIONS_NOT_MET"  # Law 38.2.1.2 direct hit from the bat


def test_runout_unconfirmed_auto_break_stays_unknown(setup):
    cam, line = setup
    o = [obs(cam, f, 2.6 - 0.15 * f, True) for f in range(3, 10)]
    auto = WicketBreak(last_intact_frame=7, last_intact_ms=140, first_broken_frame=8, first_broken_ms=160, source="auto")
    r = analyze_runout(line, o, auto, DismissalFacts(batter="non_striker", wicket_broken_fairly=True,
                                                     ball_touched_fielder_before_break=True, left_ground_to_avoid_injury=False))
    down = next(c for c in r.rule_checks if c.id == "wicket_broken")
    assert down.status == CheckStatus.UNKNOWN and r.rules_summary == "UNDETERMINED"


def test_runout_no_ball_keeper_exception(setup):
    cam, line = setup
    o = [obs(cam, f, 2.6 - 0.15 * f, True) for f in range(3, 10)]
    f = DismissalFacts(batter="striker", no_ball=True, attempting_run=False, broken_by_wicket_keeper_without_intervention=True,
                       wicket_broken_fairly=True, ball_touched_fielder_before_break=True, left_ground_to_avoid_injury=False)
    r = analyze_runout(line, o, wb(), f)
    assert next(c for c in r.rule_checks if c.id == "no_ball_keeper_exception").status == CheckStatus.FAIL
    assert r.rules_summary == "CONDITIONS_NOT_MET"


def test_stumping(setup):
    cam, line = setup
    o = [obs(cam, f, 1.6, f < 5) for f in range(3, 10)]  # foot lifted beyond the crease
    f = DismissalFacts(batter="striker", no_ball=False, attempting_run=False, broken_by_wicket_keeper_without_intervention=True,
                       wicket_broken_fairly=True, left_ground_to_avoid_injury=False)
    r = analyze_stumping(line, o, wb(), f)
    assert r.dismissal == "stumped" and r.status == CreaseStatus.OUT_OF_GROUND and r.rules_summary == "ALL_CONDITIONS_MET"
    r2 = analyze_stumping(line, o, wb(), f.model_copy(update={"no_ball": True}))
    assert r2.rules_summary == "CONDITIONS_NOT_MET"
    # same facts as run out -> Stumped takes precedence
    r3 = analyze_runout(line, o, wb(), f.model_copy(update={"ball_touched_fielder_before_break": True}))
    assert any(c.id == "stumped_precedence" and c.status == CheckStatus.FAIL for c in r3.rule_checks)


def _panning_clip(tmp_path, n=30, break_at=14, pan_px=4):
    from ml.video.ingest import write_video

    rng = np.random.default_rng(5)
    canvas = np.full((400, 700, 3), (40, 110, 40), np.uint8)
    for _ in range(400):  # crowd/advertising-like structure for feature matching, away from the stumps
        x, y = int(rng.integers(0, 700)), int(rng.integers(0, 400))
        if 300 < x < 460 and 60 < y < 280:
            continue
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(canvas, (x, y), (x + int(rng.integers(4, 25)), y + int(rng.integers(4, 25))), col, -1)
        else:
            cv2.circle(canvas, (x, y), int(rng.integers(3, 12)), col, -1)
    def frame(i):
        c = canvas.copy()
        for x in (330, 348, 366):
            cv2.rectangle(c, (x, 140), (x + 6, 260), (235, 235, 235), -1)
        cv2.rectangle(c, (329, 134), (353, 138), (60, 200, 230), -1)
        if i >= break_at:
            cv2.rectangle(c, (420, 80), (444, 84), (60, 200, 230), -1)
        else:
            cv2.rectangle(c, (350, 134), (373, 138), (60, 200, 230), -1)
        ox = 100 + i * pan_px  # camera pans right: the scene moves left in the image
        return c[40:360, ox:ox + 480].copy()
    return write_video((frame(i) for i in range(n)), tmp_path / "pan.mp4", 50)


def test_wicket_break_follows_panning_camera(tmp_path):
    from ml.runout.events import detect_wicket_break_in_video

    path = _panning_clip(tmp_path)
    roi = (220, 85, 290, 225)  # around the stumps in frame 0 (scene x 320-380 minus offset 100)
    fixed = detect_wicket_break_in_video(path, roi)
    assert not (fixed.status == "BREAK_CANDIDATE" and fixed.first_broken_frame == 14)  # pan fools a fixed ROI
    comp = detect_wicket_break_in_video(path, roi, compensate_motion=True)
    assert comp.status == "BREAK_CANDIDATE" and comp.first_broken_frame == 14, comp.reasons
    assert any("compensated" in w for w in comp.warnings)
