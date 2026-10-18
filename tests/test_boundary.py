"""Boundary geometry, Law 19 event interpretation and boundary catches (synthetic)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.boundary.events import (
    BallCrossedRope,
    BallDead,
    BallGroundContact,
    CatchCompleted,
    FielderBallContact,
    FielderGroundContact,
    FielderRopeContact,
    GroundPoint,
)
from ml.boundary.geometry import BoundaryRope, ball_rope_crossings, image_rope, measure_against_rope
from ml.boundary.rules import BoundaryFacts, BoundaryOutcome, CatchOutcome, analyze_boundary, analyze_catch_near_boundary
from ml.calibration.camera import look_at_camera
from ml.common.schemas import BallTrack, TrackPoint, TrackQuality

# image rope: horizontal line y = 100, field below (larger y)
ROPE = BoundaryRope(points=[(0, 100), (500, 100), (1000, 100)], field_side_point=(500, 600))


def gp(x, y, sigma=2.0):
    return GroundPoint(point=(x, y), sigma=sigma)


INSIDE, BEYOND, ON_ROPE = gp(400, 130), gp(400, 80), gp(400, 100.5)


# ---------------------------------------------------------------------------- geometry
def test_signed_distance_and_states():
    assert ROPE.signed_distance((300, 150)) == pytest.approx(50)
    assert ROPE.signed_distance((300, 60)) == pytest.approx(-40)
    assert measure_against_rope(ROPE, (300, 150), 2.0).state == "INSIDE"
    assert measure_against_rope(ROPE, (300, 60), 2.0).state == "BEYOND"
    assert measure_against_rope(ROPE, (300, 101), 2.0).state == "AMBIGUOUS"


def test_curved_rope_side_is_consistent():
    # arc of a circle (radius 65 m) around the field centre, traced in order
    ang = np.linspace(np.pi / 3, 2 * np.pi / 3, 9)
    rope = BoundaryRope(frame="world", points=[(65 * np.cos(a), 65 * np.sin(a)) for a in ang], field_side_point=(0.0, 0.0))
    for a in np.linspace(np.pi / 3 + 0.05, 2 * np.pi / 3 - 0.05, 7):
        assert rope.signed_distance((60 * np.cos(a), 60 * np.sin(a))) > 0
        assert rope.signed_distance((70 * np.cos(a), 70 * np.sin(a))) < 0


def test_world_rope_measured_from_calibrated_image():
    cam = look_at_camera("deep", np.array([0.0, 40.0, 12.0]), np.array([0.0, 70.0, 0.0]), 1800.0, 1920, 1080)
    rope = BoundaryRope(frame="world", points=[(-30, 70), (0, 70), (30, 70)], field_side_point=(0, 0), camera=cam)
    uv_in = cam.project(np.array([[2.0, 69.8, 0.0]]))[0]
    uv_out = cam.project(np.array([[2.0, 70.2, 0.0]]))[0]
    m_in = measure_against_rope(rope, uv_in, 1.0, "image")
    m_out = measure_against_rope(rope, uv_out, 1.0, "image")
    assert m_in.units == "m" and m_in.signed_distance == pytest.approx(0.2, abs=1e-3) and m_in.state == "INSIDE"
    assert m_out.signed_distance == pytest.approx(-0.2, abs=1e-3) and m_out.state == "BEYOND"
    assert 0 < m_in.sigma < 0.1
    img = image_rope(rope)
    assert img.frame == "image" and img.signed_distance(uv_in) > 0 > img.signed_distance(uv_out)


def test_ball_rope_crossing_uses_observations_only():
    def tp(i, y, kind="observed"):
        return TrackPoint(frame_index=i, timestamp_ms=i * 20.0, kind=kind, x=500.0, y=y)

    q = TrackQuality(num_frames_spanned=6, num_observed=4, num_missing=2, observed_ratio=0.66, longest_gap=2,
                     mean_innovation_px=1.0, mean_confidence=0.8)
    pts = [tp(0, 300), tp(1, 200), tp(2, 120, "predicted"), tp(3, 60, "interpolated"), tp(4, 40), tp(5, 30)]
    tr = BallTrack(track_id=1, points=pts, quality=q, score=1.0)
    cr = ball_rope_crossings(tr, ROPE)
    assert len(cr) == 1
    assert (cr[0].frame_before, cr[0].frame_after, cr[0].direction) == (1, 4, "outwards")
    assert "not evidence of a boundary" in cr[0].note
    only_pred = BallTrack(track_id=2, points=[tp(0, 300), tp(1, 50, "predicted")], quality=q, score=1.0)
    assert ball_rope_crossings(only_pred, ROPE) == []


# ------------------------------------------------------------------------- Law 19
def test_six_four_and_unknown_bat():
    six = [BallGroundContact(timestamp_ms=1000, position=BEYOND)]
    assert analyze_boundary(six, ROPE, BoundaryFacts(struck_by_bat=True)).status == BoundaryOutcome.BOUNDARY_SIX
    assert analyze_boundary(six, ROPE, BoundaryFacts(struck_by_bat=True)).runs_allowance == 6
    assert analyze_boundary(six, ROPE, BoundaryFacts(struck_by_bat=False)).status == BoundaryOutcome.BOUNDARY_FOUR
    r = analyze_boundary(six, ROPE)
    assert r.status == BoundaryOutcome.INCONCLUSIVE and r.boundary_scored is True
    four = [BallGroundContact(timestamp_ms=500, position=INSIDE), BallGroundContact(timestamp_ms=1000, position=BEYOND)]
    r4 = analyze_boundary(four, ROPE, BoundaryFacts(struck_by_bat=True))
    assert r4.status == BoundaryOutcome.BOUNDARY_FOUR and r4.runs_allowance == 4


def test_ball_landing_on_rope_counts_as_beyond_or_ambiguous():
    # 0.5 px inside the inner edge with 2 px sigma: cannot be called
    assert analyze_boundary([BallGroundContact(timestamp_ms=0, position=ON_ROPE)], ROPE,
                            BoundaryFacts(struck_by_bat=True)).status == BoundaryOutcome.INCONCLUSIVE


def test_ball_crossing_in_air_is_not_a_boundary():
    ev = [BallCrossedRope(timestamp_ms=100, in_air=True),
          FielderBallContact(timestamp_ms=150, fielder_id="F1", airborne=True, last_ground_contact=INSIDE),
          FielderGroundContact(timestamp_ms=300, fielder_id="F1", position=BEYOND, in_contact_with_ball=False),
          BallGroundContact(timestamp_ms=400, position=INSIDE), BallDead(timestamp_ms=2000)]
    r = analyze_boundary(ev, ROPE, BoundaryFacts(struck_by_bat=True))
    assert r.status == BoundaryOutcome.NO_BOUNDARY and r.boundary_scored is False


def test_fielder_grounded_beyond_touches_ball():
    ev = [FielderBallContact(timestamp_ms=100, fielder_id="F1", airborne=False, ground_contacts=[INSIDE, BEYOND])]
    r = analyze_boundary(ev, ROPE, BoundaryFacts(struck_by_bat=True))
    assert r.status == BoundaryOutcome.BOUNDARY_SIX  # 19.4.2: ball regarded as grounded beyond, never grounded inside
    assert r.timeline[-1].decisive and "19.4.2" in r.timeline[-1].law_reference
    rope = [FielderBallContact(timestamp_ms=100, fielder_id="F1", airborne=False, ground_contacts=[INSIDE], touching_rope=True)]
    assert analyze_boundary(rope, ROPE, BoundaryFacts(struck_by_bat=True)).boundary_scored is True


def test_bunny_hop_first_touch_from_outside_is_boundary():
    # Law 19.5.2 (2026 edition): first touch airborne after last grounding beyond -> grounded beyond
    ev = [FielderBallContact(timestamp_ms=100, fielder_id="F1", airborne=True, last_ground_contact=BEYOND)]
    assert analyze_boundary(ev, ROPE, BoundaryFacts(struck_by_bat=True)).status == BoundaryOutcome.BOUNDARY_SIX


def test_relay_after_jumping_from_outside():
    base = [FielderBallContact(timestamp_ms=100, fielder_id="F1", airborne=True, last_ground_contact=INSIDE),
            FielderGroundContact(timestamp_ms=200, fielder_id="F1", position=BEYOND, in_contact_with_ball=False),
            FielderBallContact(timestamp_ms=300, fielder_id="F1", airborne=True, last_ground_contact=BEYOND)]
    # 19.5.2.1: lands back inside -> no boundary
    ok = base + [FielderGroundContact(timestamp_ms=400, fielder_id="F1", position=INSIDE, in_contact_with_ball=False),
                 CatchCompleted(timestamp_ms=600, fielder_id="F2")]
    assert analyze_boundary(ok, ROPE, BoundaryFacts(struck_by_bat=True)).status == BoundaryOutcome.NO_BOUNDARY
    assert analyze_catch_near_boundary(ok, ROPE).status == CatchOutcome.CATCH_VALID
    # 19.5.2.1 / 19.5.2.2: lands beyond after the airborne touch -> boundary, even without the ball
    bad = base + [FielderGroundContact(timestamp_ms=400, fielder_id="F1", position=BEYOND, in_contact_with_ball=False),
                  CatchCompleted(timestamp_ms=600, fielder_id="F2")]
    assert analyze_boundary(bad, ROPE, BoundaryFacts(struck_by_bat=True)).status == BoundaryOutcome.BOUNDARY_SIX
    assert analyze_catch_near_boundary(bad, ROPE).status == CatchOutcome.CATCH_INVALID
    # touching the rope after jumping from outside is also grounding beyond
    rope = base + [FielderRopeContact(timestamp_ms=400, fielder_id="F1", in_contact_with_ball=False)]
    assert analyze_boundary(rope, ROPE, BoundaryFacts(struck_by_bat=True)).boundary_scored is True


def test_unclosed_sequence_and_unknowns_are_inconclusive():
    ev = [BallGroundContact(timestamp_ms=100, position=INSIDE)]
    assert analyze_boundary(ev, ROPE).status == BoundaryOutcome.INCONCLUSIVE
    unk = [FielderBallContact(timestamp_ms=100, fielder_id="F1", airborne=None), BallDead(timestamp_ms=500)]
    assert analyze_boundary(unk, ROPE).status == BoundaryOutcome.INCONCLUSIVE
    assert analyze_boundary([], ROPE).status == BoundaryOutcome.INSUFFICIENT_DATA


# ---------------------------------------------------------------------- Law 33 catch
def test_catch_stepping_on_rope_before_control():
    ev = [FielderBallContact(timestamp_ms=100, fielder_id="F1", airborne=False, ground_contacts=[INSIDE]),
          FielderGroundContact(timestamp_ms=250, fielder_id="F1", position=BEYOND, in_contact_with_ball=True),
          CatchCompleted(timestamp_ms=400, fielder_id="F1")]
    r = analyze_catch_near_boundary(ev, ROPE)
    assert r.status == CatchOutcome.CATCH_INVALID
    good = [ev[0], FielderGroundContact(timestamp_ms=250, fielder_id="F1", position=INSIDE, in_contact_with_ball=True), ev[2]]
    g = analyze_catch_near_boundary(good, ROPE, BoundaryFacts(no_ball=False, ball_touched_bat=True))
    assert g.status == CatchOutcome.CATCH_VALID
    assert all(c.status == "PASS" for c in g.rule_checks)


def test_catch_after_ball_touches_ground_is_invalid():
    ev = [BallGroundContact(timestamp_ms=100, position=INSIDE), CatchCompleted(timestamp_ms=200, fielder_id="F1")]
    assert analyze_catch_near_boundary(ev, ROPE).status == CatchOutcome.CATCH_INVALID


def test_catch_with_ambiguous_foot_is_inconclusive():
    ev = [FielderBallContact(timestamp_ms=100, fielder_id="F1", airborne=False, ground_contacts=[ON_ROPE]),
          CatchCompleted(timestamp_ms=400, fielder_id="F1")]
    r = analyze_catch_near_boundary(ev, ROPE)
    assert r.status == CatchOutcome.INCONCLUSIVE
    assert analyze_catch_near_boundary(ev[:1], ROPE).status == CatchOutcome.INCONCLUSIVE  # never completed
