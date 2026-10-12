import numpy as np
import pytest

from ml.calibration.camera import CalibrationError, estimate_pose, look_at_camera
from ml.calibration.pitch_geometry import PitchSpec, WicketSpec
from ml.calibration.stumps_landmarks import calibrate_from_stumps, landmarks_from_stumps
from scripts.evaluate_stumps_calibration import wicket_box

L = PitchSpec().stumps_distance_m


def _broadcast_cam(focal=9000.0):
    # Behind the bowler's end, elevated, long lens: the usual delivery view.
    return look_at_camera("tv", np.array([0.8, L + 70.0, 12.0]), np.array([0.0, 8.0, 0.5]), focal, 1920, 1080)


def _boxes(cam):
    return [[*wicket_box(cam, y0, WicketSpec()), 0.9] for y0 in (0.0, L)]


def test_far_wicket_is_striker_end_and_left_edge_is_plus_x():
    cam = _broadcast_cam()
    lm = landmarks_from_stumps(_boxes(cam)[::-1])  # order of detections must not matter
    assert lm.wickets_used == 2
    striker = [p for p in lm.points if p["end"] == "striker"]
    assert all(p["world"][1] == 0.0 for p in striker)
    off = next(p for p in striker if p["label"] == "striker_off_rh_base")
    leg = next(p for p in striker if p["label"] == "striker_leg_rh_base")
    assert off["world"][0] > 0 and off["image"][0] < leg["image"][0]


def test_two_wickets_recover_telephoto_focal_length():
    cam = _broadcast_cam(focal=12000.0)
    est, lm = calibrate_from_stumps(_boxes(cam), (1920, 1080))
    assert est is not None and not est.validated
    assert abs(est.K_np[0, 0] / 12000.0 - 1) < 0.02
    assert np.linalg.norm(est.center_world - cam.center_world) < 2.0
    assert any("stumps detector" in a for a in est.assumptions)


def test_focal_grid_escapes_mirrored_pose():
    # Low camera, long lens: the default wide-angle initial guess converges to a pose with the
    # landmarks behind the camera; the focal grid must still find the true camera.
    f = 120 * (50 + L) / WicketSpec().top_m
    cam = look_at_camera("tv", np.array([-1.5, L + 50.0, 8.0]), np.array([0.0, 4.0, 0.5]), f, 1920, 1080)
    lm = landmarks_from_stumps(_boxes(cam))
    with pytest.raises(CalibrationError):
        estimate_pose("tv", lm.world, lm.image, (1920, 1080))
    est, _ = calibrate_from_stumps(_boxes(cam), (1920, 1080))
    assert abs(est.K_np[0, 0] / f - 1) < 0.01


def test_one_wicket_gives_seeds_but_no_calibration():
    cam = _broadcast_cam()
    est, lm = calibrate_from_stumps(_boxes(cam)[:1], (1920, 1080))
    assert est is None and lm.wickets_used == 1 and len(lm.points) == 4
    assert any("Only one wicket" in w for w in lm.warnings)


def test_flat_or_weak_boxes_are_ignored():
    lm = landmarks_from_stumps([[100, 100, 200, 120, 0.9], [500, 300, 510, 380, 0.1]])
    assert lm.wickets_used == 0 and lm.warnings


def test_low_rms_but_implausible_camera_is_rejected():
    # A camera standing inside the field (e.g. boxes on a fielder's legs) fits 8 points well
    # but is not a broadcast delivery camera.
    cam = look_at_camera("tv", np.array([0.0, L + 15.0, 4.0]), np.array([0.0, 6.0, 0.5]), 2300.0, 1920, 1080)
    est, lm = calibrate_from_stumps(_boxes(cam), (1920, 1080))
    assert est is None and any("Implausible" in w for w in lm.warnings)


def test_crease_registration_recovers_camera_from_one_wicket():
    # The near stumps are hidden by an umpire; the painted creases at both ends fix the focal length.
    from ml.calibration.pitch_registration import register_pitch
    from ml.evaluation.synth_pitch import render
    from scripts.evaluate_stumps_calibration import backproject

    cam = look_at_camera("tv", np.array([0.5, L + 80.0, 12.0]), np.array([0.0, 8.0, 0.5]), 100 * (80 + L) / WicketSpec().top_m, 1920, 1080)
    reg = register_pitch(render(cam, noise=3), _boxes(cam)[:1])
    assert reg.camera is not None and reg.inlier_ratio > 0.8 and reg.near_end_inliers > 40
    ground = [np.array([x, y, 0.0]) for x in (-1.0, 1.0) for y in (1.0, 4.0, 8.0)]
    err = [np.linalg.norm(backproject(reg.camera, cam.project(X[None])[0], 2, 0.0) - X) for X in ground]
    assert np.median(err) < 0.25
    assert not reg.camera.validated


def test_auto_result_becomes_named_landmarks():
    from ml.calibration.auto import AutoCalibration, landmark_points
    from ml.calibration.pitch_geometry import standard_landmarks

    cam = _broadcast_cam()
    boxes = _boxes(cam)
    est, lm = calibrate_from_stumps(boxes, (1920, 1080))
    pts = landmark_points(AutoCalibration(est, "two_wicket_stumps", 0, boxes, lm.points), (1920, 1080))
    names = {p["name"] for p in pts}
    assert {"striker_off_stump_base_rh", "bowler_popping_crease_center"} <= names <= set(standard_landmarks())
    seeds = landmark_points(AutoCalibration(None, "seeds_only", 0, boxes[:1], landmarks_from_stumps(boxes[:1]).points), (1920, 1080))
    assert {p["name"] for p in seeds} == {"striker_off_stump_base_rh", "striker_off_stump_top_rh",
                                         "striker_leg_stump_base_rh", "striker_leg_stump_top_rh"}
    # Seeds sit on the box corners: off side on the image left in a behind-the-bowler view.
    off = next(p for p in seeds if p["name"] == "striker_off_stump_base_rh")
    assert off["image"] == [boxes[0][0], boxes[0][3]]
