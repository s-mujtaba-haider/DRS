"""Popping-crease geometry for run-out and stumping analysis (vision/geometry layer only).

Law 7.3 (MCC Laws of Cricket, 2017 Code 4th Edition, 2026): the popping crease is the
*back edge* of the crease marking, 4 ft / 1.22 m in front of the bowling crease, and is
unlimited in length. Law 30.1.1: a batter is out of their ground unless some part of their
person or bat is grounded *behind* the popping crease. A point on the white marking is
therefore not behind the crease; only the side towards the stumps counts.

Sign convention used everywhere in this package: positive signed distance = behind the
crease (towards that end's stumps), negative = on or beyond it (towards the other end).

Two ways to define the crease:
* from a calibrated ``CameraModel`` (the world line z = 0, y = popping crease of that end,
  projected into the image). A ground-contact point can then be back-projected onto z = 0
  for a metric distance in metres. This assumes the measured point is on the ground,
  which is only meaningful when the batter's bat or foot is grounded.
* from two operator-clicked image points on the back edge of the marking plus one point
  known to be behind it (e.g. a stump base). Distances are then in pixels only.

Localisation uncertainty (pixel sigma, propagated to metres through the camera) turns
near-line points into ``AMBIGUOUS`` rather than a guess.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from ml.calibration.camera import CameraModel
from ml.calibration.pitch_geometry import PitchSpec

End = Literal["striker", "bowler"]
CreaseState = Literal["BEHIND", "NOT_BEHIND", "AMBIGUOUS"]


class CreaseGeometryError(ValueError):
    pass


class CreaseLine(BaseModel):
    source: Literal["calibration", "manual"]
    end: End | None = None
    p1: tuple[float, float] = Field(description="image point on the crease back edge (px)")
    p2: tuple[float, float]
    behind_ref: tuple[float, float] = Field(description="image point known to be behind the crease (stump side)")
    camera: CameraModel | None = None
    crease_y_m: float | None = Field(None, description="pitch-frame y of the popping crease (calibration source only)")
    warnings: list[str] = Field(default_factory=list)

    def normal(self) -> np.ndarray:
        """Unit image normal pointing to the behind (stump) side."""
        a, b = np.asarray(self.p1, float), np.asarray(self.p2, float)
        d = b - a
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        if float(n @ (np.asarray(self.behind_ref, float) - a)) < 0:
            n = -n
        return n

    def signed_px(self, uv) -> float:
        return float(self.normal() @ (np.asarray(uv, float) - np.asarray(self.p1, float)))


class CreaseMeasurement(BaseModel):
    point_px: tuple[float, float]
    signed_px: float = Field(description="+ = behind the crease (stump side), image pixels")
    sigma_px: float
    signed_m: float | None = Field(None, description="+ = behind, metres along the pitch (ground-plane back-projection)")
    sigma_m: float | None = None
    ground_point_m: tuple[float, float] | None = Field(None, description="pitch-frame (x, y) of the back-projected point")
    p_behind: float = Field(description="P(point is behind the crease) under a Gaussian localisation model")
    state: CreaseState
    basis: Literal["metric", "image"]


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def crease_y(end: End, pitch: PitchSpec = PitchSpec()) -> float:
    return pitch.popping_crease_from_stumps_m if end == "striker" else pitch.stumps_distance_m - pitch.popping_crease_from_stumps_m


def crease_from_camera(camera: CameraModel, end: End, pitch: PitchSpec = PitchSpec(), half_length_m: float = 3.0) -> CreaseLine:
    """Project the popping crease of ``end`` into the image of a calibrated camera."""
    y_c = crease_y(end, pitch)
    y_stumps = 0.0 if end == "striker" else pitch.stumps_distance_m
    world = np.array([[-half_length_m, y_c, 0.0], [half_length_m, y_c, 0.0], [0.0, y_stumps, 0.0]])
    if not np.all(camera.is_in_front(world)):
        raise CreaseGeometryError(f"The {end}-end crease is not in front of camera '{camera.camera_id}'")
    uv = camera.project(world)
    warnings = []
    if not camera.validated:
        warnings.append("Camera calibration is not validated; crease position inherits its error")
    if any(abs(d) > 1e-9 for d in camera.dist):
        warnings.append("Lens distortion present: the drawn image line is a chord; metric distances use undistorted rays")
    return CreaseLine(source="calibration", end=end, p1=tuple(uv[0]), p2=tuple(uv[1]), behind_ref=tuple(uv[2]),
                      camera=camera, crease_y_m=y_c, warnings=warnings)


def crease_from_points(p1, p2, behind_point, end: End | None = None) -> CreaseLine:
    """Crease from two operator-clicked points on the back edge of the marking and a point behind it."""
    a, b, c = (np.asarray(p, float) for p in (p1, p2, behind_point))
    if np.linalg.norm(b - a) < 5.0:
        raise CreaseGeometryError("Crease points must be at least 5 px apart")
    line = CreaseLine(source="manual", end=end, p1=tuple(a), p2=tuple(b), behind_ref=tuple(c))
    if abs(line.signed_px(c)) < 2.0:
        raise CreaseGeometryError("The behind-reference point lies on the crease line; click a point clearly behind it")
    line.warnings.append("Manual crease: distances are in image pixels; perspective makes pixels non-uniform in metres")
    return line


def _ground_signed_m(line: CreaseLine, uv) -> tuple[float, tuple[float, float]] | None:
    cam = line.camera
    o, d = cam.ray(np.asarray(uv, float))
    if d[2] >= -1e-9:
        return None
    s = -o[2] / d[2]
    X = o + s * d
    behind_sign = -1.0 if line.end == "striker" else 1.0  # striker end: behind means y < crease
    return behind_sign * (X[1] - line.crease_y_m), (float(X[0]), float(X[1]))


def measure_point(line: CreaseLine, uv, sigma_px: float = 2.0, decision_confidence: float = 0.95) -> CreaseMeasurement:
    """Signed distance of an image point (bat tip / foot ground contact) to the crease, with uncertainty."""
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    uv = (float(uv[0]), float(uv[1]))
    spx = line.signed_px(uv)
    out = dict(point_px=uv, signed_px=spx, sigma_px=sigma_px)
    metric = line.camera is not None and line.crease_y_m is not None and line.end is not None
    if metric:
        g0 = _ground_signed_m(line, uv)
        if g0 is None:
            metric = False
        else:
            # numerical Jacobian of the metric distance w.r.t. pixel position
            grads = []
            for du, dv in ((1.0, 0.0), (0.0, 1.0)):
                g = _ground_signed_m(line, (uv[0] + du, uv[1] + dv))
                grads.append(np.nan if g is None else g[0] - g0[0])
            sigma_m = float(sigma_px * np.hypot(*grads)) if np.all(np.isfinite(grads)) else float("nan")
            if not np.isfinite(sigma_m) or sigma_m <= 0:
                metric = False
            else:
                out.update(signed_m=float(g0[0]), sigma_m=sigma_m, ground_point_m=g0[1])
    z = out["signed_m"] / out["sigma_m"] if metric else spx / sigma_px
    p = _phi(z)
    state: CreaseState = "BEHIND" if p >= decision_confidence else ("NOT_BEHIND" if p <= 1 - decision_confidence else "AMBIGUOUS")
    return CreaseMeasurement(**out, p_behind=p, state=state, basis="metric" if metric else "image")
