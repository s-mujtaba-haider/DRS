"""Calibration correspondences from detected stumps boxes (seeds for, not a replacement of, manual clicks).

A stumps bounding box spans the outer edges of the off and leg stumps horizontally and the ground
to the top of the bails vertically, so its four corners map to four known pitch-frame points.

Assumptions (returned with every result):
* Behind-the-bowler delivery view: the camera looks along -y, towards the striker. With two
  wickets, the smaller (far) box is the striker's end and the larger (near) box the bowler's end.
  With one wicket it is taken to be the striker's end (the near wicket is usually occluded).
* Looking along -y, world +x appears on the image left, so the left box edge is x = +half_width.
* The wicket plane is close to fronto-parallel, so box edges are taken as the stump outlines.

Four corners of one wicket are coplanar and cannot fix focal length; both wickets (8 points in two
parallel planes) are needed for the approximate-intrinsics pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.calibration.camera import CalibrationError, CameraModel, estimate_pose
from ml.calibration.pitch_geometry import PitchSpec, WicketSpec

ASSUMPTIONS = [
    "Behind-the-bowler view: far (smaller) wicket = striker's end",
    "Image-left box edge = pitch +x (camera looks along -y)",
    "Box corners taken as outer stump edges at ground and bail-top height",
]


@dataclass
class StumpsLandmarks:
    points: list[dict] = field(default_factory=list)  # {"label", "world": [x,y,z], "image": [u,v], "end"}
    wickets_used: int = 0
    assumptions: list[str] = field(default_factory=lambda: list(ASSUMPTIONS))
    warnings: list[str] = field(default_factory=list)

    @property
    def world(self) -> np.ndarray:
        return np.array([p["world"] for p in self.points], dtype=np.float64)

    @property
    def image(self) -> np.ndarray:
        return np.array([p["image"] for p in self.points], dtype=np.float64)


def landmarks_from_stumps(
    boxes, min_score: float = 0.25, min_aspect: float = 1.5,
    pitch: PitchSpec = PitchSpec(), wicket: WicketSpec = WicketSpec(),
) -> StumpsLandmarks:
    """boxes: iterable of (x1, y1, x2, y2, score) in pixels. Uses the two best upright boxes."""
    cand = [tuple(map(float, b)) for b in boxes]
    cand = [b for b in cand if b[4] >= min_score and (b[3] - b[1]) >= min_aspect * (b[2] - b[0])]
    cand = sorted(cand, key=lambda b: -b[4])
    out = StumpsLandmarks()
    if not cand:
        out.warnings.append("No upright stumps box above the score threshold")
        return out
    # Behind the bowler, the near wicket is taller and its base clearly lower in the image. Pairs
    # that violate this (duplicate boxes on one wicket, a box on a player) cannot be two wickets.
    # A base gap of 1.5 near-wicket heights rejects stacked boxes on one wicket (top and bottom
    # halves, seen on real clips); it also refuses ~13% of genuine pairs from low (< ~6 m) cameras.
    pair = next(((f, n) for i, a in enumerate(cand) for b in cand[i + 1:]
                 for f, n in [sorted((a, b), key=lambda q: q[3] - q[1])]
                 if (n[3] - n[1]) >= 1.05 * (f[3] - f[1]) and n[3] - f[3] >= 1.5 * (n[3] - n[1])), None)
    if pair:
        far, near = pair
        ends = [("striker", 0.0, far), ("bowler", pitch.stumps_distance_m, near)]
    else:
        if len(cand) > 1:
            out.warnings.append("Several stumps boxes, but no pair is consistent with two wickets in a behind-the-bowler view")
        ends = [("striker", 0.0, cand[0])]
        out.warnings.append("Only one wicket found: its end is assumed (striker); 4 coplanar points cannot fix focal length")
    hw, top = wicket.half_width_m, wicket.top_m
    for end, y0, (x1, y1, x2, y2, _) in ends:
        for side, x_img, x_w in (("off_rh", x1, hw), ("leg_rh", x2, -hw)):
            out.points.append({"label": f"{end}_{side}_base", "world": [x_w, y0, 0.0], "image": [x_img, y2], "end": end})
            out.points.append({"label": f"{end}_{side}_bail_top", "world": [x_w, y0, top], "image": [x_img, y1], "end": end})
    out.wickets_used = len(ends)
    return out


def calibrate_from_stumps(boxes, image_size: tuple[int, int], camera_id: str = "main", **kw) -> tuple[CameraModel | None, StumpsLandmarks]:
    """Approximate-intrinsics pose from both wickets; None (with the landmarks as seeds) otherwise."""
    lm = landmarks_from_stumps(boxes, **kw)
    if lm.wickets_used < 2:
        return None, lm
    # Broadcast delivery views are long telephoto shots (focal 2-15x the image width): a single
    # wide-angle initial guess converges to a mirrored pose, so start from a log-spaced grid.
    cam, last_err = None, None
    for f0 in np.geomspace(1.0, 16.0, 9) * max(image_size):
        try:
            c = estimate_pose(camera_id, lm.world, lm.image, image_size, focal_guess_px=float(f0))
        except CalibrationError as exc:
            last_err = exc
            continue
        if cam is None or c.reprojection_rms_px < cam.reprojection_rms_px:
            cam = c
    if cam is None:
        raise last_err
    # Eight points fit almost any camera to ~1 px, so a low RMS is no evidence that the boxes were
    # really the two wickets. A broadcast delivery camera sits beyond the boundary, elevated, close
    # to the pitch axis; anything else means a false pair (e.g. a fielder's legs) and is rejected.
    x, y, z = cam.center_world
    if not (y - kw.get("pitch", PitchSpec()).stumps_distance_m / 2 >= 45.0 and 3.0 <= z <= 60.0 and abs(x) <= 15.0):
        lm.warnings.append(f"Implausible broadcast camera position ({x:.1f}, {y:.1f}, {z:.1f}) m: "
                           "boxes are probably not the two wickets")
        return None, lm
    cam.assumptions = [*cam.assumptions, *lm.assumptions, "Correspondences from an automatic stumps detector"]
    cam.warnings = [*cam.warnings, *lm.warnings]
    cam.validated = False
    return cam, lm
