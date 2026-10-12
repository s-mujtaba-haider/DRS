import numpy as np
import pytest
from pydantic import ValidationError

from ml.calibration.camera import CalibrationError, CameraModel, estimate_pose, look_at_camera
from ml.calibration.pitch_geometry import standard_landmarks
from ml.reconstruction.triangulation import CameraObservation, associate_and_triangulate, triangulate_dlt

LANDMARK_NAMES = [
    "striker_off_stump_base_rh", "striker_leg_stump_base_rh", "striker_off_stump_top_rh", "striker_leg_stump_top_rh",
    "striker_popping_crease_x_plus_1m32", "striker_popping_crease_x_minus_1m32", "striker_bowling_crease_right",
    "striker_bowling_crease_left", "striker_middle_stump_top", "bowler_off_stump_top_rh", "bowler_leg_stump_top_rh",
]


def _cams():
    a = look_at_camera("A", np.array([0.5, 30.0, 6.0]), np.array([0, 2, 0.5]), 1500, 1920, 1080)
    b = look_at_camera("B", np.array([25.0, 3.0, 4.0]), np.array([0, 2, 0.5]), 2200, 1920, 1080)
    c = look_at_camera("C", np.array([-20.0, -10.0, 5.0]), np.array([0, 2, 0.5]), 2000, 1920, 1080)
    for cam in (a, b, c):
        cam.sync_uncertainty_ms = 0.5
    return a, b, c


def test_coordinate_projection_roundtrip():
    cam, *_ = _cams()
    X = np.array([[0.1, 2.0, 0.5]])
    uv = cam.project(X)
    o, d = cam.ray(uv[0])
    # The ray passes through X.
    s = np.dot(X[0] - o, d)
    assert np.linalg.norm(o + s * d - X[0]) < 1e-6


def test_pose_with_known_intrinsics_recovers_camera():
    cam, *_ = _cams()
    lm = standard_landmarks()
    W = np.array([lm[n] for n in LANDMARK_NAMES])
    uv = cam.project(W)
    est = estimate_pose("A", W, uv, (1920, 1080), K=cam.K_np, intrinsics_validated=True)
    assert est.reprojection_rms_px < 1e-3
    assert np.allclose(est.center_world, cam.center_world, atol=1e-3)
    assert est.validated


def test_approximate_mode_estimates_focal_and_is_flagged():
    cam, *_ = _cams()
    lm = standard_landmarks()
    W = np.array([lm[n] for n in LANDMARK_NAMES])
    uv = cam.project(W) + np.random.default_rng(0).normal(0, 0.5, (len(W), 2))
    est = estimate_pose("A", W, uv, (1920, 1080))
    assert est.calibration_type == "approximate_focal_from_landmarks"
    assert not est.validated
    assert any("Principal point" in a for a in est.assumptions)
    assert est.K[0][0] == pytest.approx(1500, rel=0.05)


@pytest.mark.parametrize(
    "world,image,msg",
    [
        (np.zeros((3, 3)), np.zeros((3, 2)), "At least 4"),
        (np.random.rand(5, 3), np.random.rand(4, 2), "differ"),
        (np.random.rand(5, 3), np.full((5, 2), -10.0), "inside the image"),
    ],
)
def test_calibration_input_validation(world, image, msg):
    with pytest.raises(CalibrationError, match=msg):
        estimate_pose("x", world, image, (100, 100), K=np.eye(3))


def test_poor_calibration_flagged():
    cam, *_ = _cams()
    lm = standard_landmarks()
    W = np.array([lm[n] for n in LANDMARK_NAMES])
    uv = cam.project(W)
    uv[0] += [25, -20]  # mislabelled point
    est = estimate_pose("A", W, uv, (1920, 1080), K=cam.K_np, intrinsics_validated=True)
    assert est.reprojection_rms_px > 3
    assert not est.validated
    assert any("unreliable" in w for w in est.warnings)


def test_camera_model_schema_validation():
    with pytest.raises(ValidationError):
        CameraModel(camera_id="x", image_width=10, image_height=10, K=[[1, 0], [0, 1]], R=np.eye(3).tolist(),
                    t=[0, 0, 0], calibration_type="imported")


def test_triangulation_exact_synthetic():
    a, b, _ = _cams()
    X = np.array([0.2, 3.0, 0.8])
    est = triangulate_dlt([a.P, b.P], [a.project(X)[0], b.project(X)[0]])
    assert np.allclose(est, X, atol=1e-6)


def test_associate_and_triangulate_with_sync_offset_and_outlier():
    a, b, c = _cams()
    b.time_offset_ms = 5.0  # camera B clock runs 5 ms behind
    obs = []
    truth = []
    for k in range(10):
        t = k * 4.0
        X = np.array([0.1, 10 - k * 0.8, 1.0 + 0.05 * k])
        truth.append(X)
        for cam, offset in ((a, 0.0), (b, 5.0), (c, 0.0)):
            u, v = cam.project(X)[0]
            if cam.camera_id == "C" and k == 4:
                u += 60  # inconsistent observation in one view
            obs.append(CameraObservation(camera_id=cam.camera_id, frame_index=k, timestamp_ms=t - offset, u=u, v=v, sigma_px=1.0))
    res = associate_and_triangulate([a, b, c], obs, sync_tolerance_ms=1.0)
    assert len(res.points) == 10
    for p, X in zip(res.points, truth):
        assert np.allclose(p.xyz, X, atol=1e-3)
    assert res.points[4].rejected_cameras == ["C"]
    assert res.points[0].quality == "good"
    assert not res.calibration_validated


def test_unsynchronised_cameras_do_not_associate():
    a, b, _ = _cams()
    X = np.array([0.0, 5.0, 1.0])
    obs = [CameraObservation(camera_id="A", frame_index=0, timestamp_ms=0.0, u=a.project(X)[0][0], v=a.project(X)[0][1]),
           CameraObservation(camera_id="B", frame_index=0, timestamp_ms=20.0, u=b.project(X)[0][0], v=b.project(X)[0][1])]
    res = associate_and_triangulate([a, b], obs, sync_tolerance_ms=2.0)
    assert res.points == [] and res.unmatched_observations == 2


def test_triangulation_needs_two_cameras():
    a, *_ = _cams()
    with pytest.raises(ValueError, match="two"):
        associate_and_triangulate([a], [])
