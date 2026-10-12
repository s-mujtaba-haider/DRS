"""Register the pitch model (stumps + creases at both ends) to a broadcast frame.

Why: two detected wickets calibrate a camera, but in delivery views the near wicket is usually
hidden by the umpire or bowler. One wicket (4 coplanar points) fixes the camera pose for any
assumed focal length but cannot fix the focal length itself. The painted creases can: the
near-end popping and return creases are usually visible even when the near stumps are not.

Method:
1. White-line mask of the frame (``pitch_lines.white_line_mask``) and its distance transform.
2. For a log-spaced grid of focal lengths, solve the pose from the stumps corners (one or two
   wickets) with that focal length; keep only plausible broadcast cameras.
3. Score each camera by projecting points sampled along every crease of the pitch model and
   averaging the truncated distance to the nearest white pixel (a chamfer score).
4. Refine the best focal length on a finer grid, then refine focal length and pose together
   (stumps corners + crease chamfer, robust loss).

The result reports the crease inlier ratio (projected crease points within 2 px of a white
pixel) as evidence; it is an automatic seed and is never marked validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ml.calibration.camera import CameraModel
from ml.calibration.pitch_geometry import PitchSpec, WicketSpec
from ml.calibration.pitch_lines import white_line_mask
from ml.calibration.stumps_landmarks import ASSUMPTIONS, StumpsLandmarks, landmarks_from_stumps

POPPING_HALF_LENGTH_M = 1.83  # the popping crease extends at least 6 ft either side of the middle stump
RETURN_BEHIND_STUMPS_M = 1.22  # return creases run back to 8 ft behind the popping crease


def crease_model(pitch: PitchSpec = PitchSpec(), step_m: float = 0.04) -> np.ndarray:
    """Points sampled along the painted creases at both ends (pitch frame, metres)."""
    L, bc, pc = pitch.stumps_distance_m, pitch.bowling_crease_half_length_m, pitch.popping_crease_from_stumps_m
    segs = []
    for y0, s in ((0.0, 1.0), (L, -1.0)):
        segs.append(((-POPPING_HALF_LENGTH_M, y0 + s * pc), (POPPING_HALF_LENGTH_M, y0 + s * pc)))  # popping
        segs.append(((-bc, y0), (bc, y0)))  # bowling
        for x in (-bc, bc):  # return creases
            segs.append(((x, y0 - s * RETURN_BEHIND_STUMPS_M), (x, y0 + s * pc)))
    pts = []
    for (xa, ya), (xb, yb) in segs:
        n = max(2, int(math.hypot(xb - xa, yb - ya) / step_m))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([xa + t * (xb - xa), ya + t * (yb - ya), np.zeros(n)], 1))
    return np.concatenate(pts)


@dataclass
class Registration:
    camera: CameraModel | None
    landmarks: StumpsLandmarks
    chamfer_px: float | None = None  # mean truncated distance of visible crease points
    inlier_ratio: float | None = None  # visible crease points within 2 px of a white pixel
    visible_points: int = 0
    recall_px: float | None = None  # white pixels in the projected pitch area -> nearest crease
    near_end_inliers: int = 0  # bowler-end crease points on white: the evidence that fixes focal length
    stumps_rms_px: float | None = None
    accepted: bool = False
    warnings: list[str] = field(default_factory=list)


def _camera(K: np.ndarray, rvec, tvec, size, cid: str) -> CameraModel:
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    return CameraModel(camera_id=cid, image_width=size[0], image_height=size[1], K=K.tolist(), dist=[0.0] * 5,
                       R=R.tolist(), t=np.asarray(tvec, float).ravel().tolist(),
                       calibration_type="approximate_focal_from_landmarks", validated=False)


def _K(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)


def _project(K, rvec, tvec, X):
    uv, _ = cv2.projectPoints(X, np.asarray(rvec, float), np.asarray(tvec, float), K, None)
    return uv.reshape(-1, 2)


def _krt(f, w, h, xc, yc, zc, yaw, pitch_down, roll):
    """Camera at (xc, yc, zc) looking towards -y, turned by yaw (about z), tilted down by pitch, rolled."""
    d = np.array([math.sin(yaw) * math.cos(pitch_down), -math.cos(yaw) * math.cos(pitch_down), -math.sin(pitch_down)])
    x = np.cross(d, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    R = np.vstack([x, y, d])
    c, s = math.cos(roll), math.sin(roll)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    t = -R @ np.array([xc, yc, zc])
    rvec, _ = cv2.Rodrigues(R)
    return _K(f, w, h), rvec.ravel(), t


def _project_params(f, w, h, xc, yc, zc, yaw, pitch_down, roll, X):
    return _project(*_krt(f, w, h, xc, yc, zc, yaw, pitch_down, roll), X)


def _plausible(rvec, tvec, pitch: PitchSpec) -> bool:
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    x, y, z = (-R.T @ np.asarray(tvec, float).ravel())
    # Beyond the boundary but not absurdly far: a very distant, very long-lens camera can collapse
    # both ends of the pitch onto one set of creases.
    return 45.0 <= y - pitch.stumps_distance_m / 2 <= 180.0 and 3.0 <= z <= 60.0 and abs(x) <= 15.0


class _Scorer:
    def __init__(self, mask: np.ndarray, model: np.ndarray, trunc_px: float, exclude_boxes):
        self.mask = mask
        self.dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        self.h, self.w = mask.shape
        self.model = model
        self.trunc = trunc_px
        self.offframe_penalty = 2.0  # px per unit fraction of model points out of view
        self.exclude = [tuple(map(float, b[:4])) for b in exclude_boxes]

    def residuals(self, K, rvec, tvec) -> tuple[np.ndarray, np.ndarray]:
        R, _ = cv2.Rodrigues(np.asarray(rvec, float))
        zc = (self.model @ R.T + np.asarray(tvec, float).ravel())[:, 2]
        uv = _project(K, rvec, tvec, self.model)
        vis = (zc > 0) & (uv[:, 0] >= 1) & (uv[:, 0] < self.w - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < self.h - 1)
        for x1, y1, x2, y2 in self.exclude:  # the stumps themselves occlude the bowling crease
            vis &= ~((uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2 + 2))
        d = np.full(len(uv), self.trunc)
        p = uv[vis]
        d[vis] = np.minimum(cv2.remap(self.dt, p[:, 0].astype(np.float32).reshape(-1, 1),
                                      p[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel(), self.trunc)
        return d, vis

    def recall(self, K, rvec, tvec, pitch: PitchSpec = PitchSpec(), trunc_px: float = 6.0) -> float:
        """Mean truncated distance from white pixels inside the projected pitch area to the projected
        creases (half resolution). Penalises cameras that leave visible creases unexplained."""
        s = 0.5
        hs, ws = int(self.h * s), int(self.w * s)
        L = pitch.stumps_distance_m
        poly = np.array([[-POPPING_HALF_LENGTH_M, -RETURN_BEHIND_STUMPS_M, 0], [POPPING_HALF_LENGTH_M, -RETURN_BEHIND_STUMPS_M, 0],
                         [POPPING_HALF_LENGTH_M, L + RETURN_BEHIND_STUMPS_M, 0], [-POPPING_HALF_LENGTH_M, L + RETURN_BEHIND_STUMPS_M, 0]])
        R, _ = cv2.Rodrigues(np.asarray(rvec, float))
        if np.any((poly @ R.T + np.asarray(tvec, float).ravel())[:, 2] <= 0):
            return trunc_px
        area = np.zeros((hs, ws), np.uint8)
        cv2.fillPoly(area, [(_project(K, rvec, tvec, poly) * s).astype(np.int32)], 255)
        white = cv2.resize(self.mask, (ws, hs), interpolation=cv2.INTER_NEAREST) & area
        for x1, y1, x2, y2 in self.exclude:
            white[int(y1 * s):int(y2 * s) + 2, int(x1 * s):int(x2 * s) + 1] = 0
        ys, xs = np.nonzero(white)
        if len(xs) < 20:
            return trunc_px
        canvas = np.full((hs, ws), 255, np.uint8)
        for p in (_project(K, rvec, tvec, self.model) * s).astype(np.int32):
            if 0 <= p[0] < ws and 0 <= p[1] < hs:
                canvas[p[1], p[0]] = 0
        dt = cv2.distanceTransform(canvas, cv2.DIST_L2, 3) / s
        return float(np.minimum(dt[ys, xs], trunc_px).mean())

    def score(self, K, rvec, tvec, min_visible: int = 60) -> tuple[float, int]:
        d, vis = self.residuals(K, rvec, tvec)
        n = int(vis.sum())
        if n < min_visible:
            return float("inf"), n
        # Mean distance over the visible crease points, plus a mild penalty for the share pushed
        # out of view (parts of the model are legitimately off-frame, e.g. near-end return creases
        # running towards the camera), so the search cannot win by framing the model away.
        return float(d[vis].mean()) + self.offframe_penalty * (1.0 - n / len(d)), n


def register_pitch(frame: np.ndarray, boxes, camera_id: str = "main", min_score: float = 0.25,
                   focal_range=(1.0, 24.0), trunc_px: float = 6.0, pitch: PitchSpec = PitchSpec(),
                   wicket: WicketSpec = WicketSpec()) -> Registration:
    h, w = frame.shape[:2]
    lm = landmarks_from_stumps(boxes, min_score=min_score, pitch=pitch, wicket=wicket)
    reg = Registration(camera=None, landmarks=lm, warnings=list(lm.warnings))
    if lm.wickets_used == 0:
        return reg
    Wp, Ip = lm.world.astype(np.float64), lm.image.astype(np.float64)
    used_boxes = [b for b in boxes if b[4] >= min_score]
    scorer = _Scorer(white_line_mask(frame, kernel_px=max(9, int(0.012 * w)) | 1), crease_model(pitch), trunc_px, used_boxes)
    from scipy.optimize import least_squares

    # A stumps box is axis-aligned, so it carries no perspective lean: one wicket leaves both the
    # focal length and the camera height free. Search both; for each pair fit the level camera
    # (no roll) to the stumps corners, then let the creases decide.
    far = min((b for b in used_boxes), key=lambda b: b[3] - b[1])
    far_h = far[3] - far[1]

    def fit_pose(f, z):
        D = f * wicket.top_m / far_h  # distance to the far wicket from its apparent height
        y_c = math.sqrt(max(D * D - z * z, 1.0))
        # Lateral position and yaw are nearly interchangeable for one wicket; broadcast cameras sit
        # close to the pitch axis, so x_c starts at 0 here and only the creases may move it.
        x0 = np.array([y_c, 0.0, math.atan2(z, y_c)])  # y_c, yaw, pitch(down)
        def r(p):
            return (_project_params(f, w, h, 0.0, p[0], z, p[1], p[2], 0.0, Wp) - Ip).ravel()
        sol = least_squares(r, x0, method="lm", max_nfev=200)
        return np.concatenate([[0.0], sol.x]), float(np.sqrt(np.mean(sol.fun ** 2)))

    # Box corners are only approximately the stump corners, so the stumps alone leave the pose a few
    # pixels off; the creases refine the pose for each (focal, height) cell, with a wide truncation
    # (12 px) so the basin of attraction covers that error. Stumps residuals are down-weighted.
    wide = _Scorer(scorer.mask, scorer.model, 12.0, used_boxes)

    def refine_pose(f, z, p):
        def r(q):
            xc, yc, yaw, pit, roll = q
            r_st = 0.5 * (_project_params(f, w, h, xc, yc, z, yaw, pit, roll, Wp) - Ip).ravel()
            d, _ = wide.residuals(*_krt(f, w, h, xc, yc, z, yaw, pit, roll))
            return np.concatenate([r_st, 0.3 * d, [xc / 3.0, roll / 0.02]])  # priors: near the axis, level
        sol = least_squares(r, [p[0], p[1], p[2], p[3], 0.0], diff_step=1e-5, max_nfev=60)
        return sol.x

    results = []
    for f in np.geomspace(*focal_range, 40) * w:
        for z in np.geomspace(4.0, 45.0, 11):
            p, rms = fit_pose(f, z)
            if rms > 3.0:
                continue
            K, rvec, tvec = _krt(f, w, h, p[0], p[1], z, p[2], p[3], 0.0)
            if not _plausible(rvec, tvec, pitch) or not np.isfinite(wide.score(K, rvec, tvec)[0]):
                continue
            q = refine_pose(f, z, p)
            K, rvec, tvec = _krt(f, w, h, q[0], q[1], z, q[2], q[3], q[4])
            sc, n = scorer.score(K, rvec, tvec)
            if np.isfinite(sc) and _plausible(rvec, tvec, pitch):
                results.append((sc, f, z, q))
    if not results:
        reg.warnings.append("No focal length and height give a plausible broadcast camera with the creases in view")
        return reg
    results.sort(key=lambda r: r[0])
    # Re-rank the best candidates by precision (creases on white) + recall (white explained).
    head = []
    for sc0, f0, z0, q0 in results[:15]:
        Kc, rc, tc = _krt(f0, w, h, q0[0], q0[1], z0, q0[2], q0[3], q0[4])
        head.append((sc0 + scorer.recall(Kc, rc, tc, pitch), f0, z0, q0))
    results = sorted(head, key=lambda r: r[0])

    # Joint refinement from the best few starts: log-focal, camera position, yaw, pitch, roll.
    def resid(q, sc_):
        f_, xc, yc, zc, yaw, pit, roll = math.exp(q[0]), *q[1:]
        r_st = (_project_params(f_, w, h, xc, yc, zc, yaw, pit, roll, Wp) - Ip).ravel()
        d, _ = sc_.residuals(*_krt(f_, w, h, xc, yc, zc, yaw, pit, roll))
        return np.concatenate([r_st, 0.5 * d, [xc / 3.0, roll / 0.02]])

    best = None
    for _sc0, f0, z0, p0 in results[:6]:
        q0 = np.array([math.log(f0), p0[0], p0[1], z0, p0[2], p0[3], p0[4]])
        q = q0
        try:  # coarse (12 px truncation, wide basin) then fine (6 px)
            for sc_ in (wide, scorer):
                q = least_squares(resid, q, args=(sc_,), loss="soft_l1", f_scale=2.0, diff_step=1e-4, max_nfev=200).x
        except ValueError:
            pass
        K, rvec, tvec = _krt(math.exp(q[0]), w, h, *q[1:])
        sc, n = scorer.score(K, rvec, tvec)
        if not (np.isfinite(sc) and _plausible(rvec, tvec, pitch)):
            K, rvec, tvec = _krt(f0, w, h, p0[0], p0[1], z0, p0[2], p0[3], p0[4])
            sc, n = scorer.score(K, rvec, tvec)
        sc = sc + scorer.recall(K, rvec, tvec, pitch)
        if best is None or sc < best[0]:
            best = (sc, K, rvec, tvec, n)
    sc, K, rvec, tvec, n = best

    cam = _camera(K, rvec, tvec, (w, h), camera_id)
    d, vis = scorer.residuals(K, rvec, tvec)
    reg.camera = cam
    reg.chamfer_px = float(d[vis].mean()) if vis.any() else None
    reg.inlier_ratio = float((d[vis] <= 2.0).mean()) if vis.any() else 0.0
    reg.visible_points = int(vis.sum())
    reg.stumps_rms_px = float(np.sqrt(np.mean(np.sum((_project(K, rvec, tvec, Wp) - Ip) ** 2, 1))))
    reg.recall_px = scorer.recall(K, rvec, tvec, pitch)
    near = scorer.model[:, 1] > pitch.stumps_distance_m / 2
    reg.near_end_inliers = int((vis & near & (d <= 2.0)).sum())
    cam.reprojection_rms_px = reg.stumps_rms_px
    cam.num_correspondences = len(Wp)
    cam.assumptions = [
        "Principal point assumed at image centre", "Square pixels assumed", "Lens distortion assumed to be zero",
        *ASSUMPTIONS, "Focal length chosen by registering the painted creases (both ends) to white image lines",
        "Correspondences from an automatic stumps detector",
    ]
    cam.warnings = [*reg.warnings, f"Crease registration: {reg.inlier_ratio:.0%} of {reg.visible_points} visible crease points "
                    f"within 2 px of a white line (mean truncated distance {reg.chamfer_px:.2f} px)"]
    return reg
