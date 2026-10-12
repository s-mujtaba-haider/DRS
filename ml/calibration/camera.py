"""Camera models and calibration.

Supported workflows:
* intrinsic calibration from chessboard images (validated intrinsics),
* pose (extrinsic) estimation from pitch landmark correspondences via PnP,
* an explicitly *approximate* mode when intrinsics are unknown: the focal length
  is estimated from the landmarks under stated assumptions (principal point at
  image centre, square pixels, no lens distortion).

No camera parameters are invented: every CameraModel records how it was obtained,
its reprojection error and the assumptions it relies on.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

CalibrationType = Literal["chessboard_intrinsics+pnp", "pnp_known_intrinsics", "approximate_focal_from_landmarks", "imported"]


class CalibrationError(ValueError):
    pass


class CameraModel(BaseModel):
    camera_id: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    K: list[list[float]]
    dist: list[float] = Field(default_factory=lambda: [0.0] * 5)
    R: list[list[float]]
    t: list[float]
    calibration_type: CalibrationType
    validated: bool = Field(False, description="True only when intrinsics came from a physical calibration target")
    reprojection_rms_px: float | None = None
    num_correspondences: int | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    time_offset_ms: float = Field(0.0, description="Add to this camera's timestamps to get the common clock")
    sync_uncertainty_ms: float | None = None
    world_frame: str = "pitch"

    @field_validator("K", "R")
    @classmethod
    def _square3(cls, v):
        if len(v) != 3 or any(len(r) != 3 for r in v):
            raise ValueError("must be a 3x3 matrix")
        return v

    @field_validator("t")
    @classmethod
    def _vec3(cls, v):
        if len(v) != 3:
            raise ValueError("t must have 3 elements")
        return v

    @property
    def K_np(self) -> np.ndarray:
        return np.asarray(self.K, dtype=float)

    @property
    def R_np(self) -> np.ndarray:
        return np.asarray(self.R, dtype=float)

    @property
    def t_np(self) -> np.ndarray:
        return np.asarray(self.t, dtype=float)

    @property
    def P(self) -> np.ndarray:
        return self.K_np @ np.hstack([self.R_np, self.t_np[:, None]])

    @property
    def center_world(self) -> np.ndarray:
        return -self.R_np.T @ self.t_np

    def project(self, X: np.ndarray, distort: bool = True) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        rvec, _ = cv2.Rodrigues(self.R_np)
        d = np.asarray(self.dist, dtype=float) if distort else np.zeros(5)
        pts, _ = cv2.projectPoints(X, rvec, self.t_np, self.K_np, d)
        return pts.reshape(-1, 2)

    def undistort_points(self, uv: np.ndarray) -> np.ndarray:
        """Pixel coords with lens distortion removed (still in pixels)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float)).reshape(-1, 1, 2)
        if not any(self.dist):
            return uv.reshape(-1, 2)
        return cv2.undistortPoints(uv, self.K_np, np.asarray(self.dist), P=self.K_np).reshape(-1, 2)

    def ray(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """World-frame ray (origin, unit direction) through an image point."""
        u, v = self.undistort_points(np.asarray(uv))[0]
        d_cam = np.linalg.solve(self.K_np, np.array([u, v, 1.0]))
        d = self.R_np.T @ d_cam
        return self.center_world, d / np.linalg.norm(d)

    def is_in_front(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return ((self.R_np @ X.T).T + self.t_np)[:, 2] > 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CameraModel:
        return cls.model_validate(json.loads(Path(path).read_text()))


def calibrate_intrinsics_chessboard(
    images: list[np.ndarray], pattern_size: tuple[int, int], square_size_m: float
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Standard Zhang calibration. Returns K, dist, rms, number of usable views."""
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_size_m
    obj_pts, img_pts, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern_size)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            obj_pts.append(objp)
            img_pts.append(corners)
    if len(obj_pts) < 3:
        raise CalibrationError(f"Chessboard found in only {len(obj_pts)} image(s); at least 3 are required")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return K, dist.ravel(), float(rms), len(obj_pts)


def _validate_correspondences(world: np.ndarray, image: np.ndarray, width: int, height: int) -> list[str]:
    warnings = []
    if world.shape[0] != image.shape[0]:
        raise CalibrationError("Number of world and image points differ")
    if world.shape[0] < 4:
        raise CalibrationError("At least 4 correspondences are required for pose estimation")
    if world.shape[1] != 3 or image.shape[1] != 2:
        raise CalibrationError("World points must be 3D and image points 2D")
    if not np.all(np.isfinite(world)) or not np.all(np.isfinite(image)):
        raise CalibrationError("Correspondences contain non-finite values")
    if np.any(image[:, 0] < 0) or np.any(image[:, 0] > width) or np.any(image[:, 1] < 0) or np.any(image[:, 1] > height):
        raise CalibrationError("Image points must lie inside the image")
    if len({tuple(p) for p in world.round(6)}) != len(world):
        raise CalibrationError("Duplicate world points")
    # Planarity check: smallest singular value of centred points.
    s = np.linalg.svd(world - world.mean(0), compute_uv=False)
    if s[-1] < 1e-3 * s[0]:
        warnings.append("All landmarks are coplanar: depth along the plane normal is weakly constrained")
    spread = np.ptp(image, axis=0)
    if spread[0] < 0.1 * width and spread[1] < 0.1 * height:
        warnings.append("Image points cover a small region of the frame: pose will be poorly conditioned")
    return warnings


def estimate_pose(
    camera_id: str,
    world_points: np.ndarray,
    image_points: np.ndarray,
    image_size: tuple[int, int],
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
    intrinsics_validated: bool = False,
    max_rms_px: float = 3.0,
) -> CameraModel:
    width, height = image_size
    world = np.asarray(world_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    warnings = _validate_correspondences(world, image, width, height)
    assumptions: list[str] = []

    if K is None:
        if len(world) < 6:
            raise CalibrationError("Approximate mode (unknown intrinsics) needs at least 6 correspondences")
        assumptions += [
            "Principal point assumed at image centre",
            "Square pixels assumed",
            "Lens distortion assumed to be zero",
            "Focal length estimated from pitch landmarks, not from a calibration target",
        ]
        f0 = 1.2 * max(width, height)
        K0 = np.array([[f0, 0, width / 2], [0, f0, height / 2], [0, 0, 1]], dtype=np.float64)
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        )
        try:
            _, K_est, _, rvecs, tvecs = cv2.calibrateCamera(
                [world.astype(np.float32)], [image.astype(np.float32)], (width, height), K0, np.zeros(5), flags=flags
            )
        except cv2.error as exc:
            raise CalibrationError(f"Focal-length estimation failed: {exc}") from exc
        K_use, dist_use = K_est, np.zeros(5)
        rvec, tvec = rvecs[0], tvecs[0]
        ctype: CalibrationType = "approximate_focal_from_landmarks"
        validated = False
    else:
        K_use = np.asarray(K, dtype=np.float64)
        dist_use = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
        # SQPnP is globally optimal (no initial guess), avoiding the mirrored-pose local
        # minima that the iterative solver can fall into with a noisy or mislabelled point.
        ok, rvec, tvec = cv2.solvePnP(world, image, K_use, dist_use, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise CalibrationError("solvePnP failed")
        rvec, tvec = cv2.solvePnPRefineLM(world, image, K_use, dist_use, rvec, tvec)
        ctype = "chessboard_intrinsics+pnp" if intrinsics_validated else "pnp_known_intrinsics"
        validated = intrinsics_validated
        if not intrinsics_validated:
            assumptions.append("Intrinsics supplied by user but not marked as validated")

    R, _ = cv2.Rodrigues(rvec)
    cam = CameraModel(
        camera_id=camera_id, image_width=width, image_height=height, K=K_use.tolist(), dist=list(map(float, dist_use)),
        R=R.tolist(), t=tvec.ravel().tolist(), calibration_type=ctype, validated=validated,
        num_correspondences=len(world), assumptions=assumptions, warnings=warnings,
    )
    if not np.all(cam.is_in_front(world)):
        raise CalibrationError("Estimated pose places some landmarks behind the camera: check point labels")
    err = np.linalg.norm(cam.project(world) - image, axis=1)
    cam.reprojection_rms_px = float(math.sqrt(np.mean(err**2)))
    if cam.reprojection_rms_px > max_rms_px:
        cam.warnings.append(
            f"Reprojection RMS {cam.reprojection_rms_px:.2f}px exceeds {max_rms_px}px: calibration is unreliable"
        )
        cam.validated = False
    if len(world) <= 6:
        cam.warnings.append("Few correspondences: reprojection error is not an independent accuracy check")
    return cam


def look_at_camera(
    camera_id: str, position: np.ndarray, target: np.ndarray, focal_px: float, width: int, height: int
) -> CameraModel:
    """Construct a synthetic camera (used by tests and simulations only; flagged as imported)."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    z = target - position
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ position
    K = [[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1]]
    return CameraModel(camera_id=camera_id, image_width=width, image_height=height, K=K, R=R.tolist(), t=t.tolist(),
                       calibration_type="imported", validated=False, assumptions=["Synthetic camera"])
