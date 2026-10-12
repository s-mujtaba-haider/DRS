"""Pitch coordinate system and standard wicket/pitch dimensions.

Coordinate system ("pitch frame", metres, right-handed):
* origin: ground level at the base of the middle stump, striker's (batting) end
* +y: along the pitch towards the bowler's end
* +z: vertically up
* +x: completes a right-handed frame, i.e. to the right of an observer standing at
  the striker's wicket looking towards the bowler (= the bowler's left-hand side)

A right-handed batter stands with the left shoulder towards the bowler and faces
the off side, so for a right-hander the off side is +x; for a left-hander it is -x.
(This matches the cricket-synth dataset frame up to a shift of half the pitch length.)

Dimensions follow the MCC Laws of Cricket (Law 7 The pitch, Law 8 The wickets,
Law 4 The ball; men's senior cricket). Values are configurable in configs/lbw.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

INCH = 0.0254


@dataclass(frozen=True)
class WicketSpec:
    width_m: float = 9 * INCH  # 22.86 cm, outer edges of outer stumps
    stump_height_m: float = 28 * INCH  # 71.12 cm above ground
    bail_height_above_stumps_m: float = 0.5 * INCH  # 1.27 cm, bails project above stump tops
    stump_diameter_m: float = 1.375 * INCH  # Law 8.2: between 1 3/8 and 1 1/2 in

    @property
    def top_m(self) -> float:
        return self.stump_height_m + self.bail_height_above_stumps_m

    @property
    def half_width_m(self) -> float:
        return self.width_m / 2.0


@dataclass(frozen=True)
class PitchSpec:
    stumps_distance_m: float = 22 * 0.9144  # 20.12 m between the two sets of stumps
    pitch_width_m: float = 10 * 0.3048  # 3.05 m (5 ft each side of middle stump)
    popping_crease_from_stumps_m: float = 4 * 0.3048  # 1.22 m
    bowling_crease_half_length_m: float = 4 * 0.3048 + 4 * INCH  # 8 ft 8 in total => 1.32 m half
    return_crease_from_middle_m: float = 4 * 0.3048 + 4 * INCH  # 4 ft 4 in = 1.32 m


@dataclass(frozen=True)
class BallSpec:
    # Law 4: men's ball circumference 22.4-22.9 cm -> radius ~3.57-3.64 cm
    radius_m: float = 0.229 / (2 * 3.141592653589793)


def standard_landmarks(pitch: PitchSpec = PitchSpec(), wicket: WicketSpec = WicketSpec()) -> dict[str, tuple[float, float, float]]:
    """Named world points usable as calibration correspondences."""
    L, hw, top = pitch.stumps_distance_m, wicket.half_width_m, wicket.stump_height_m
    pc = pitch.popping_crease_from_stumps_m
    bc = pitch.bowling_crease_half_length_m
    pts: dict[str, tuple[float, float, float]] = {}
    for end, y0, sign in (("striker", 0.0, 1.0), ("bowler", L, -1.0)):
        pts[f"{end}_off_stump_base_rh"] = (hw, y0, 0.0)
        pts[f"{end}_leg_stump_base_rh"] = (-hw, y0, 0.0)
        pts[f"{end}_middle_stump_base"] = (0.0, y0, 0.0)
        pts[f"{end}_off_stump_top_rh"] = (hw, y0, top)
        pts[f"{end}_leg_stump_top_rh"] = (-hw, y0, top)
        pts[f"{end}_middle_stump_top"] = (0.0, y0, top)
        pts[f"{end}_bowling_crease_right"] = (bc, y0, 0.0)
        pts[f"{end}_bowling_crease_left"] = (-bc, y0, 0.0)
        pts[f"{end}_popping_crease_center"] = (0.0, y0 + sign * pc, 0.0)
        pts[f"{end}_popping_crease_x_plus_1m32"] = (bc, y0 + sign * pc, 0.0)
        pts[f"{end}_popping_crease_x_minus_1m32"] = (-bc, y0 + sign * pc, 0.0)
    return pts
