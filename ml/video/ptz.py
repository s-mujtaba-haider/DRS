"""Broadcast camera motion within a shot: pan, tilt and zoom about a fixed optical centre.

Broadcast delivery cameras zoom in during the delivery and follow the play. A calibration made on
one frame is then wrong for the others. For a camera rotating and zooming about a fixed centre,
every frame is related to the reference frame by a homography (H = K_j R K_ref^-1), so image
points can be mapped into the reference frame's pixel coordinates exactly, and the reference
calibration applies to them. (The lens entrance pupil moves slightly with zoom; at broadcast
distances of 60-150 m this is negligible.)

Homographies are estimated between consecutive frames from ORB features with a robust (MAGSAC)
fit and chained to the reference frame. Static on-screen graphics (score bar, logos) do not move
with the camera and would pull the estimate towards identity, so pixels whose intensity barely
changes over the segment while the camera moves are masked out. Moving players are rejected as
outliers by the robust fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ShotMotion:
    ref_frame: int
    to_ref: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> 3x3 homography, frame px -> reference px
    inliers: dict[int, int] = field(default_factory=dict)  # frame -> inlier matches to its neighbour
    warnings: list[str] = field(default_factory=list)

    def zoom(self, frame: int) -> float | None:
        """Magnification of ``frame`` relative to the reference frame (>1: ``frame`` is zoomed in further)."""
        H = self.to_ref.get(frame)
        if H is None:
            return None
        Hn = H / H[2, 2]
        return float(1.0 / np.sqrt(abs(np.linalg.det(Hn[:2, :2]))))

    def map_points(self, frame: int, uv: np.ndarray) -> np.ndarray | None:
        H = self.to_ref.get(frame)
        if H is None:
            return None
        p = cv2.perspectiveTransform(np.asarray(uv, np.float64).reshape(-1, 1, 2), H)
        return p.reshape(-1, 2)


def _overlay_mask(grays: list[np.ndarray], min_std: float = 2.0) -> np.ndarray:
    """255 where the image changes over the segment (scene), 0 on static graphics."""
    if len(grays) < 3:
        return np.full(grays[0].shape, 255, np.uint8)
    stack = np.stack(grays[:: max(1, len(grays) // 12)], 0).astype(np.float32)
    std = stack.std(0)
    static = (std < min_std).astype(np.uint8)
    static = cv2.dilate(static, np.ones((9, 9), np.uint8))
    return np.where(static > 0, 0, 255).astype(np.uint8)


def estimate_shot_motion(frames: dict[int, np.ndarray], ref_frame: int, n_features: int = 3000,
                         min_inliers: int = 40) -> ShotMotion:
    """Homographies mapping each frame of a contiguous, cut-free segment into ``ref_frame``.

    ``frames`` maps frame index -> BGR image and must contain consecutive indices around
    ``ref_frame``. Frames whose chain to the reference breaks (too few inliers) are left out.
    """
    idx = sorted(frames)
    out = ShotMotion(ref_frame=ref_frame)
    if ref_frame not in frames:
        out.warnings.append("Reference frame not in the segment")
        return out
    grays = {i: cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in idx}
    mask = _overlay_mask([grays[i] for i in idx])
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    feats = {i: orb.detectAndCompute(grays[i], mask) for i in idx}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def pair(a: int, b: int) -> tuple[np.ndarray | None, int]:
        """Homography mapping frame a -> frame b."""
        (ka, da), (kb, db) = feats[a], feats[b]
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return None, 0
        m = matcher.match(da, db)
        if len(m) < 8:
            return None, 0
        pa = np.float32([ka[x.queryIdx].pt for x in m])
        pb = np.float32([kb[x.trainIdx].pt for x in m])
        H, inl = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 2.0, maxIters=5000, confidence=0.999)
        n = int(inl.sum()) if inl is not None else 0
        return (H if n >= min_inliers else None), n

    out.to_ref[ref_frame] = np.eye(3)
    for direction in (1, -1):  # walk away from the reference in both directions
        cur, H_cur = ref_frame, np.eye(3)
        while (nxt := cur + direction) in frames:
            H, n = pair(nxt, cur)  # nxt -> cur
            out.inliers[nxt] = n
            if H is None:
                out.warnings.append(f"Camera-motion chain broken at frame {nxt} ({n} inliers): "
                                    f"frames beyond it are not mapped")
                break
            H_cur = H_cur @ H  # nxt -> cur -> ... -> ref
            H_cur = H_cur / H_cur[2, 2]
            out.to_ref[nxt] = H_cur
            cur = nxt
    return out


def read_segment(video: str, start: int, end: int) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = {}
    for i in range(start, end + 1):
        ok, f = cap.read()
        if not ok:
            break
        out[i] = f
    cap.release()
    return out


@dataclass
class StabilisedObservations:
    uv: np.ndarray  # positions in the reference frame's pixels
    radius: np.ndarray  # apparent radius rescaled to the reference frame's zoom
    keep: np.ndarray  # observations that could be mapped
    notes: list[str] = field(default_factory=list)


def stabilise_observations(video: str, frames: np.ndarray, uv: np.ndarray, radius: np.ndarray, ref_frame: int) -> StabilisedObservations:
    """Map ball observations from their own frames into ``ref_frame`` (the calibration frame).

    The reference calibration then applies to every observation even though the broadcast camera
    panned and zoomed between frames. Apparent radii scale with the local zoom. Observations whose
    frame cannot be chained to the reference (a cut in between, or too few feature matches) are
    dropped and reported.
    """
    frames = np.asarray(frames, int)
    lo, hi = int(min(frames.min(), ref_frame)), int(max(frames.max(), ref_frame))
    motion = estimate_shot_motion(read_segment(video, lo, hi), ref_frame)
    out_uv, out_r, keep = np.array(uv, float), np.array(radius, float), np.zeros(len(frames), bool)
    zooms = []
    for k, f in enumerate(frames):
        H = motion.to_ref.get(int(f))
        if H is None:
            continue
        out_uv[k] = motion.map_points(int(f), uv[k][None])[0]
        scale = float(np.sqrt(abs(np.linalg.det((H / H[2, 2])[:2, :2]))))  # frame -> reference magnification
        out_r[k] = radius[k] * scale
        keep[k] = True
        zooms.append(1.0 / scale)
    notes = [f"Camera pan/tilt/zoom compensated: {int(keep.sum())}/{len(frames)} observations mapped into the "
             f"calibration frame {ref_frame}"
             + (f" (camera zoom over the observations {min(zooms):.2f}x-{max(zooms):.2f}x of the calibration frame)" if zooms else "")]
    if not keep.all():
        notes.append(f"{int((~keep).sum())} observation(s) dropped: their frames could not be related to the calibration frame "
                     "(shot cut or too few feature matches)")
    notes.extend(motion.warnings)
    return StabilisedObservations(out_uv, out_r, keep, notes)
