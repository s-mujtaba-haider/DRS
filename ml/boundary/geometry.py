"""Boundary geometry: the boundary as a polyline, signed distance with uncertainty (vision/geometry layer).

Law 19.2 (MCC Laws, 2017 Code 4th Edition, 2026): for a white line the boundary is the edge
nearest the pitch (19.2.2.1); for an object in contact with the ground (a rope) it is the edge
of the grounded part nearest the pitch (19.2.3). Laws 19.4.1 / 19.5.1: touching the boundary
or the marking object counts as grounded beyond. The operator should therefore trace the
*inner* (pitch-side) edge of the rope. A point on that edge is beyond, not inside.

Sign convention: positive signed distance = inside the field of play, negative = beyond.

A rope can be given in image pixels (``frame="image"``) or as ground points in the pitch frame
in metres (``frame="world"``, needs a ``CameraModel`` to measure image points, which are
back-projected onto the ground plane z = 0, so they must be ground contacts).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ml.calibration.camera import CameraModel
from ml.common.schemas import BallTrack

RopeState = Literal["INSIDE", "BEYOND", "AMBIGUOUS"]
OBSERVED_KINDS = ("observed", "manual")


class BoundaryRope(BaseModel):
    frame: Literal["image", "world"] = "image"
    points: list[tuple[float, float]] = Field(min_length=2, description="inner edge of the rope, in order")
    field_side_point: tuple[float, float] = Field(description="a point known to be inside the field (same frame)")
    camera: CameraModel | None = None

    @field_validator("points")
    @classmethod
    def _distinct(cls, v):
        if any(math.dist(a, b) < 1e-6 for a, b in zip(v, v[1:])):
            raise ValueError("consecutive rope points must differ")
        return v

    def _raw(self, p: np.ndarray) -> tuple[float, int]:
        """Unsigned distance from p to the polyline and the nearest segment index."""
        P = np.asarray(self.points, float)
        best, seg = np.inf, 0
        for k in range(len(P) - 1):
            a, b = P[k], P[k + 1]
            d = b - a
            s = np.clip(float((p - a) @ d / (d @ d)), 0.0, 1.0)
            dist = float(np.linalg.norm(p - (a + s * d)))
            if dist < best:
                best, seg = dist, k
        return best, seg

    def _side(self, p: np.ndarray, seg: int) -> float:
        P = np.asarray(self.points, float)
        a, b = P[seg], P[seg + 1]
        return float(np.sign((b - a)[0] * (p - a)[1] - (b - a)[1] * (p - a)[0]))

    def signed_distance(self, p) -> float:
        """+ inside the field, - beyond (in the rope's frame units). Assumes the field is on one consistent side."""
        p = np.asarray(p, float)
        fp = np.asarray(self.field_side_point, float)
        _, fseg = self._raw(fp)
        inside_sign = self._side(fp, fseg)
        if inside_sign == 0:
            raise ValueError("field_side_point lies on the rope")
        d, seg = self._raw(p)
        return d * (1.0 if self._side(p, seg) == inside_sign else -1.0)


class RopeMeasurement(BaseModel):
    point: tuple[float, float]
    point_frame: Literal["image", "world"]
    signed_distance: float = Field(description="+ inside the field, - beyond (units below)")
    sigma: float
    units: Literal["px", "m"]
    p_inside: float
    state: RopeState


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _to_ground(cam: CameraModel, uv) -> np.ndarray | None:
    o, d = cam.ray(np.asarray(uv, float))
    if d[2] >= -1e-9:
        return None
    return (o + (-o[2] / d[2]) * d)[:2]


def measure_against_rope(rope: BoundaryRope, point, sigma: float, point_frame: Literal["image", "world"] = "image",
                         decision_confidence: float = 0.95) -> RopeMeasurement:
    """Signed distance of a ground-contact point to the rope with a Gaussian localisation model.

    ``sigma`` is in the point's units (px for image points, m for world points).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    p = np.asarray(point, float)
    if rope.frame == point_frame:
        sd, sig, units = rope.signed_distance(p), sigma, ("px" if point_frame == "image" else "m")
    elif rope.frame == "world" and point_frame == "image":
        if rope.camera is None:
            raise ValueError("a world-frame rope needs a camera to measure image points")
        g = _to_ground(rope.camera, p)
        if g is None:
            raise ValueError("image point does not intersect the ground plane")
        sd = rope.signed_distance(g)
        grads = []
        for du in ((1.0, 0.0), (0.0, 1.0)):
            gg = _to_ground(rope.camera, p + np.asarray(du))
            grads.append(rope.signed_distance(gg) - sd if gg is not None else np.nan)
        sig = float(sigma * np.hypot(*grads))
        if not np.isfinite(sig) or sig <= 0:
            raise ValueError("cannot propagate pixel uncertainty to the ground at this point")
        units = "m"
    else:
        raise ValueError("measuring a world point against an image rope is not supported; project the point first")
    p_in = _phi(sd / sig)
    state: RopeState = "INSIDE" if p_in >= decision_confidence else ("BEYOND" if p_in <= 1 - decision_confidence else "AMBIGUOUS")
    return RopeMeasurement(point=(float(p[0]), float(p[1])), point_frame=point_frame, signed_distance=float(sd), sigma=float(sig),
                           units=units, p_inside=p_in, state=state)


def image_rope(rope: BoundaryRope, samples_per_segment: int = 10) -> BoundaryRope:
    """Image-frame copy of a world rope (projected through its camera)."""
    if rope.frame == "image":
        return rope
    if rope.camera is None:
        raise ValueError("a world-frame rope needs a camera to project")
    P = np.asarray(rope.points, float)
    dense = [P[k] + s * (P[k + 1] - P[k]) for k in range(len(P) - 1) for s in np.linspace(0, 1, samples_per_segment, endpoint=False)]
    dense.append(P[-1])
    W = np.column_stack([np.asarray(dense), np.zeros(len(dense))])
    if not np.all(rope.camera.is_in_front(W)):
        raise ValueError("part of the rope is behind the camera")
    uv = rope.camera.project(W)
    # inside reference: 1 m into the field from the middle of the rope (the given field point may be off-screen)
    k = (len(P) - 1) // 2
    a, b = P[k], P[k + 1]
    n = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
    mid = (a + b) / 2
    ref = mid + n if rope.signed_distance(mid + n) > 0 else mid - n
    if not rope.camera.is_in_front(np.array([[*ref, 0.0]]))[0]:
        raise ValueError("the field side of the rope is behind the camera")
    f = rope.camera.project(np.array([[*ref, 0.0]]))[0]
    return BoundaryRope(frame="image", points=[tuple(map(float, q)) for q in uv], field_side_point=(float(f[0]), float(f[1])))


class RopeCrossing(BaseModel):
    t_start_ms: float = Field(description="last observed time on the starting side")
    t_end_ms: float = Field(description="first observed time on the other side")
    frame_before: int
    frame_after: int
    direction: Literal["outwards", "inwards"]
    note: str = ("Image-space crossing of the traced rope line. A ball in the air projects across the rope without "
                 "being grounded beyond it; this is not evidence of a boundary on its own.")


def ball_rope_crossings(track: BallTrack, rope: BoundaryRope, sigma_px: float = 2.0, decision_confidence: float = 0.95) -> list[RopeCrossing]:
    """Crossings of the (image-space) rope line by observed ball positions only.

    Filtered, interpolated and predicted points are ignored; a crossing is reported between the
    two observed points that bracket it, so long gaps give wide time windows.
    """
    r = image_rope(rope)
    pts = [p for p in sorted(track.points, key=lambda q: q.timestamp_ms) if p.kind in OBSERVED_KINDS]
    states = []
    for p in pts:
        m = measure_against_rope(r, (p.x, p.y), max(sigma_px, p.sigma_x or 0.0), "image", decision_confidence)
        if m.state != "AMBIGUOUS":
            states.append((p, m.state))
    out = []
    for (pa, sa), (pb, sb) in zip(states, states[1:]):
        if sa != sb:
            out.append(RopeCrossing(t_start_ms=pa.timestamp_ms, t_end_ms=pb.timestamp_ms, frame_before=pa.frame_index,
                                    frame_after=pb.frame_index, direction="outwards" if sa == "INSIDE" else "inwards"))
    return out
