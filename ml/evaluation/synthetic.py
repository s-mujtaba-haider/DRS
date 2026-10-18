"""Synthetic delivery generator with exact ground truth.

Used for deterministic tests and pipeline smoke tests. Synthetic results verify
that the implementation behaves as designed; they say nothing about accuracy on
real cricket footage.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ml.calibration.camera import CameraModel, look_at_camera
from ml.calibration.pitch_geometry import BallSpec, PitchSpec, standard_landmarks
from ml.trajectory.ballistic import FlightModel


@dataclass
class SyntheticDelivery:
    model: FlightModel
    times: np.ndarray
    xyz: np.ndarray
    camera: CameraModel
    uv: np.ndarray
    impact_time: float


def broadcast_camera(width: int = 1280, height: int = 720, focal: float = 2600.0) -> CameraModel:
    """Elevated camera behind the bowler's end looking down the pitch (typical broadcast angle)."""
    return look_at_camera("synthetic_main", np.array([0.3, 32.0, 6.0]), np.array([0.0, 2.0, 0.4]), focal, width, height)


def make_delivery(
    pitch_y: float = 4.5,
    line_x: float = 0.0,
    end_x: float = 0.0,
    speed_mps: float = 33.0,
    release_height: float = 2.2,
    post_bounce_vz: float = 3.0,
    fps: float = 50.0,
    impact_y: float = 1.2,
    camera: CameraModel | None = None,
    ball_radius: float = BallSpec().radius_m,
    drag_c: float = 0.0,
) -> SyntheticDelivery:
    """Construct a delivery that pitches at (line_x, pitch_y) and passes x=end_x at the stumps (y=0).

    ``drag_c`` (1/m, quadratic drag as in ``FlightModel``; ``ballistic.DEFAULT_DRAG_C`` is a real
    ball) adds air drag. The launch and post-bounce velocities are then solved numerically so the
    delivery still pitches at (line_x, pitch_y) and reaches x = end_x at the stumps. With the default
    0 the flight is drag-free (the original behaviour, kept for existing tests).
    """
    L = PitchSpec().stumps_distance_m
    p0 = np.array([0.15, L - 1.2, release_height])
    # Time to reach the pitching point at roughly constant along-pitch speed.
    t_b = (p0[1] - pitch_y) / speed_mps
    vy = -speed_mps
    vx = (line_x - p0[0]) / t_b
    # z(t_b) = r  =>  vz = (r - z0 + 0.5 g t^2) / t
    vz = (ball_radius - p0[2] + 0.5 * 9.80665 * t_b**2) / t_b
    v0 = np.array([vx, vy, vz])
    t_after = pitch_y / (speed_mps * 0.9)
    v1 = np.array([(end_x - line_x) / t_after, -speed_mps * 0.9, post_bounce_vz])
    model = FlightModel(np.concatenate([p0, v0, v1]), 0.0, True, ball_radius)
    if drag_c > 0:
        model = _solve_with_drag(p0, v0, v1, ball_radius, drag_c, line_x, pitch_y, end_x)
    # Impact: when the ball reaches y = impact_y.
    ts = np.arange(0, 1.5, 1e-4)
    ys = model.position(ts)[:, 1]
    impact_time = float(ts[np.argmax(ys <= impact_y)])
    times = np.arange(0.0, impact_time + 1e-9, 1.0 / fps)
    xyz = model.position(times)
    cam = camera or broadcast_camera()
    uv = cam.project(xyz)
    return SyntheticDelivery(model, times, xyz, cam, uv, impact_time)


def _solve_with_drag(p0, v0, v1, r, drag_c, line_x, pitch_y, end_x) -> FlightModel:
    """Adjust (vx0, vy0, vz0, vx1) so the dragged flight pitches at (line_x, pitch_y) and passes end_x at y = 0."""
    from scipy.optimize import least_squares

    def build(q):
        th = np.concatenate([p0, [q[0], q[1], q[2]], [q[3], v1[1], v1[2]]])
        return FlightModel(th, 0.0, True, r, drag_c=drag_c)

    def resid(q):
        m = build(q)
        tb = m.bounce_time()
        if tb is None:
            return np.array([10.0, 10.0, 10.0])
        pb = m.position(np.array([tb]))[0]
        ts = np.arange(tb, tb + 1.0, 2e-4)
        ys = m.position(ts)
        k = int(np.argmax(ys[:, 1] <= 0.0)) if np.any(ys[:, 1] <= 0.0) else len(ts) - 1
        return np.array([pb[0] - line_x, pb[1] - pitch_y, ys[k, 0] - end_x])

    q0 = np.array([v0[0], v0[1] * 1.04, v0[2], v1[0]])  # drag slows the ball: start slightly faster
    sol = least_squares(resid, q0, x_scale=[1, 5, 1, 1], xtol=1e-10, ftol=1e-10)
    if np.max(np.abs(sol.fun)) > 5e-3:
        raise ValueError(f"could not solve dragged delivery to targets (residual {np.max(np.abs(sol.fun)):.3f} m)")
    return build(sol.x)


def render_video_frames(d: SyntheticDelivery, lead_frames: int = 10, tail_frames: int = 10, noise: float = 4.0,
                        seed: int = 0):
    """Yield (frame_bgr, ball_uv_or_None) for a simple rendered scene with pitch and stumps."""
    rng = np.random.default_rng(seed)
    cam = d.camera
    w, h = cam.image_width, cam.image_height
    base = np.zeros((h, w, 3), np.uint8)
    base[:] = (60, 120, 50)  # outfield green (BGR)
    ps = PitchSpec()
    corners = np.array([[-ps.pitch_width_m / 2, -2, 0], [ps.pitch_width_m / 2, -2, 0],
                        [ps.pitch_width_m / 2, ps.stumps_distance_m + 2, 0], [-ps.pitch_width_m / 2, ps.stumps_distance_m + 2, 0]])
    cv2.fillPoly(base, [cam.project(corners).astype(np.int32)], (120, 170, 200))
    lm = standard_landmarks()
    for end in ("striker", "bowler"):
        for side in ("off", "leg"):
            a, b = lm[f"{end}_{side}_stump_base_rh"], lm[f"{end}_{side}_stump_top_rh"]
            pa, pb = cam.project(np.array([a, b]))
            cv2.line(base, tuple(map(int, pa)), tuple(map(int, pb)), (230, 230, 230), 2)
        a, b = lm[f"{end}_middle_stump_base"], lm[f"{end}_middle_stump_top"]
        pa, pb = cam.project(np.array([a, b]))
        cv2.line(base, tuple(map(int, pa)), tuple(map(int, pb)), (230, 230, 230), 2)
        pc = cam.project(np.array([lm[f"{end}_popping_crease_x_plus_1m32"], lm[f"{end}_popping_crease_x_minus_1m32"]]))
        cv2.line(base, tuple(map(int, pc[0])), tuple(map(int, pc[1])), (255, 255, 255), 2)

    def noisy(img):
        n = rng.normal(0, noise, img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    for _ in range(lead_frames):
        yield noisy(base), None
    for uv, X in zip(d.uv, d.xyz):
        img = base.copy()
        # Apparent radius from distance to camera.
        depth = float((cam.R_np @ X + cam.t_np)[2])
        r_px = max(2, int(round(cam.K_np[0, 0] * d.model.ball_radius / depth)))
        cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), r_px, (40, 40, 200), -1, cv2.LINE_AA)
        yield noisy(img), (float(uv[0]), float(uv[1]))
    for _ in range(tail_frames):
        yield noisy(base), None


def write_synthetic_video(path: str, d: SyntheticDelivery, fps: float = 50.0, **kw) -> list[tuple[float, float] | None]:
    from ml.video.ingest import write_video

    truth: list = []

    def gen():
        for img, uv in render_video_frames(d, **kw):
            truth.append(uv)
            yield img

    write_video(gen(), path, fps)
    return truth
