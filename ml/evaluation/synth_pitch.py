"""Synthetic broadcast frame (grass, pitch strip, white creases, stumps, an occluding umpire) for
testing pitch registration. Not photorealistic: real frames add players, shadows, worn creases,
footmarks and compression."""
import cv2
import numpy as np

from ml.calibration.pitch_geometry import PitchSpec, WicketSpec
from ml.calibration.pitch_registration import crease_model


def render(cam, w=1920, h=1080, line_px=3, noise=0.0, seed=0, occlude_near=True):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (60, 130, 70)  # grass (BGR)
    p = PitchSpec()
    L = p.stumps_distance_m
    strip = np.array([[-1.6, -3, 0], [1.6, -3, 0], [1.6, L + 3, 0], [-1.6, L + 3, 0]], float)
    if np.all(cam.is_in_front(strip)):
        cv2.fillPoly(img, [cam.project(strip).astype(np.int32)], (120, 170, 190))
    pts = crease_model(p, step_m=0.01)
    vis = cam.is_in_front(pts)
    for u, v in cam.project(pts[vis]).astype(int):
        cv2.circle(img, (u, v), max(1, line_px // 2), (245, 245, 245), -1)
    wk = WicketSpec()
    for y0 in (0.0, L):
        for x in (-wk.half_width_m, 0.0, wk.half_width_m):
            a, b = cam.project(np.array([[x, y0, 0], [x, y0, wk.top_m]]))
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (40, 160, 220), 2)
    if occlude_near:  # an umpire standing in front of the near stumps
        a, b = cam.project(np.array([[-0.4, L + 1.0, 0.0], [0.4, L + 1.0, 1.9]]))
        cv2.rectangle(img, (int(a[0]), int(b[1])), (int(b[0]), int(a[1])), (30, 30, 30), -1)
    if noise:
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img
