"""Boundary (Law 19) and fair-catch-near-the-boundary (Law 33) interpretation of an event sequence.

Source: MCC Laws of Cricket, 2017 Code 4th Edition (2026), https://www.lords.org/mcc/the-laws-of-cricket
(the 2026 edition includes the revised Law 19.5.2 that ICC adopted in its playing conditions from June 2025).
* 19.4.1: the ball is grounded beyond the boundary if it touches the boundary, a marking object, the
  ground beyond, or an object grounded beyond.
* 19.4.2: it is also grounded beyond if a fielder grounded beyond touches it, or a fielder who caught it
  within the boundary becomes grounded beyond while in contact with it before completing the catch.
* 19.5.1: a fielder is grounded beyond if any part of their person touches the boundary, a marking object,
  the ground beyond, or an object in contact with the ground beyond (or, with intent, such a fielder).
* 19.5.2: an airborne fielder is grounded beyond if, before their *first* contact with the ball, their final
  ground contact was not entirely within the boundary. Applies to any fielder, whether or not another
  fielder touched the ball first.
* 19.5.2.1: if the first contact complied, the fielder may jump from outside to touch the ball while airborne;
  after that, every ground contact by that fielder until the ball is dead must be within the field of play,
  otherwise a boundary is scored. 19.5.2.2: the same applies when such a fielder returns the ball to play.
* 19.7.1: Boundary 6 if and only if the ball was struck by the bat and is first grounded beyond without having
  touched the ground within the field (even if it touched a fielder). 19.7.2: otherwise Boundary 4.
* 19.6.2: allowances 6 and 4 unless agreed otherwise before the toss.
* 33.1 / 33.2.1 / 33.3: a catch must be held before the ball touches the ground; neither the ball nor any
  fielder in contact with it may be grounded beyond the boundary before the catch is completed, which is
  when the fielder has complete control over the ball and their own movement.

The ball crossing the rope in the air is not by itself a boundary (33.2.2.4). Unobserved contacts
cannot be excluded, so a NO_BOUNDARY outcome assumes the event list is complete up to the ball becoming
dead. Not an umpiring decision.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from ml.boundary.events import (
    BallCrossedRope,
    BallDead,
    BallGroundContact,
    BoundaryEvent,
    CatchCompleted,
    FielderBallContact,
    FielderGroundContact,
    FielderRopeContact,
    GroundPoint,
)
from ml.boundary.geometry import BoundaryRope, RopeMeasurement, measure_against_rope
from ml.lbw.rules import CheckStatus, RuleCheck

ENGINE_VERSION = "boundary-engine-0.1.0-experimental"
RULES_SOURCE = "MCC Laws of Cricket, 2017 Code 4th Edition (2026): Laws 19 and 33"
DISCLAIMER = ("Experimental research output, not an official umpiring decision. Real-world accuracy of this "
              "system has not been established.")


class BoundaryOutcome(StrEnum):
    BOUNDARY_SIX = "BOUNDARY_SIX"
    BOUNDARY_FOUR = "BOUNDARY_FOUR"
    NO_BOUNDARY = "NO_BOUNDARY"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class CatchOutcome(StrEnum):
    CATCH_VALID = "CATCH_VALID"
    CATCH_INVALID = "CATCH_INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class BoundaryConfig(BaseModel):
    decision_confidence: float = Field(0.95, ge=0.5, lt=1.0)
    default_sigma_px: float = Field(3.0, gt=0)
    default_sigma_m: float = Field(0.03, gt=0)
    boundary_6_allowance: int = 6
    boundary_4_allowance: int = 4


class BoundaryFacts(BaseModel):
    struck_by_bat: bool | None = Field(None, description="Law 19.7.1: needed to tell a 6 from a 4")
    no_ball: bool | None = None  # catch analysis only
    ball_touched_bat: bool | None = None  # catch analysis only (Law 33.1)


class TimelineItem(BaseModel):
    index: int
    timestamp_ms: float
    frame_index: int | None
    kind: str
    description: str
    law_reference: str | None = None
    measurements: list[RopeMeasurement] = Field(default_factory=list)
    decisive: bool = False


class _Sim(BaseModel):
    """Internal outcome of walking the event sequence."""

    grounded_beyond_at: int | None = None  # event index where the ball became grounded beyond
    grounded_inside_before: bool = False  # ball touched the ground inside before that
    ground_inside_at: int | None = None
    ambiguous_at: int | None = None
    catch_at: int | None = None
    dead_at: int | None = None
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timeline: list[TimelineItem] = Field(default_factory=list)


def _measure(rope: BoundaryRope, gp: GroundPoint, cfg: BoundaryConfig) -> RopeMeasurement:
    sigma = gp.sigma or (cfg.default_sigma_px if gp.frame == "image" else cfg.default_sigma_m)
    return measure_against_rope(rope, gp.point, sigma, gp.frame, cfg.decision_confidence)


def _simulate(events: list[BoundaryEvent], rope: BoundaryRope, cfg: BoundaryConfig, stop_at_catch: bool) -> _Sim:
    sim = _Sim()
    ev = sorted(events, key=lambda e: e.timestamp_ms)
    touched: set[str] = set()
    jumped_outside: set[str] = set()

    def item(i, e, desc, law=None, ms=(), decisive=False):
        sim.timeline.append(TimelineItem(index=i, timestamp_ms=e.timestamp_ms, frame_index=e.frame_index, kind=e.kind,
                                         description=desc, law_reference=law, measurements=list(ms), decisive=decisive))

    def beyond(i, e, desc, law, ms=()):
        sim.grounded_beyond_at = i
        sim.reasons.append(f"{desc} at {e.timestamp_ms:.1f} ms ({law})")
        item(i, e, desc, law, ms, decisive=True)

    def ambiguous(i, e, desc, law=None, ms=()):
        sim.ambiguous_at = i
        sim.reasons.append(f"{desc} at {e.timestamp_ms:.1f} ms: cannot be resolved from the evidence")
        item(i, e, desc, law, ms, decisive=True)

    for i, e in enumerate(ev):
        if isinstance(e, BallGroundContact):
            if e.position is None:
                ambiguous(i, e, "Ball touched the ground at an unrecorded position", "Law 19.4.1")
                break
            m = _measure(rope, e.position, cfg)
            if m.state == "BEYOND":
                beyond(i, e, "Ball grounded beyond the boundary", "Law 19.4.1", [m])
                break
            if m.state == "AMBIGUOUS":
                ambiguous(i, e, "Ball touched the ground too close to the boundary to call", "Law 19.4.1", [m])
                break
            sim.grounded_inside_before = True
            sim.ground_inside_at = sim.ground_inside_at if sim.ground_inside_at is not None else i
            item(i, e, "Ball touched the ground inside the field of play", "Law 19.7.2", [m])
            if stop_at_catch:
                break
        elif isinstance(e, FielderBallContact):
            first = e.fielder_id not in touched
            touched.add(e.fielder_id)
            if e.touching_rope:
                beyond(i, e, f"Fielder {e.fielder_id} touched the ball while in contact with the boundary", "Law 19.4.2 / 19.5.1")
                break
            if e.airborne is None:
                ambiguous(i, e, f"Whether fielder {e.fielder_id} was airborne at the touch is unknown", "Law 19.5")
                break
            if not e.airborne:
                if not e.ground_contacts:
                    ambiguous(i, e, f"Ground contact of fielder {e.fielder_id} at the touch was not recorded", "Law 19.5.1")
                    break
                ms = [_measure(rope, g, cfg) for g in e.ground_contacts]
                if any(m.state == "BEYOND" for m in ms):
                    beyond(i, e, f"Fielder {e.fielder_id} touched the ball while grounded beyond the boundary", "Law 19.4.2 / 19.5.1", ms)
                    break
                if any(m.state == "AMBIGUOUS" for m in ms):
                    ambiguous(i, e, f"Fielder {e.fielder_id}'s ground contact is too close to the boundary to call", "Law 19.5.1", ms)
                    break
                item(i, e, f"Fielder {e.fielder_id} touched the ball while grounded inside", "Law 19.5.1", ms)
                continue
            # airborne touch
            if e.last_ground_contact is None:
                ambiguous(i, e, f"Airborne fielder {e.fielder_id}'s last ground contact before the touch was not recorded", "Law 19.5.2")
                break
            m = _measure(rope, e.last_ground_contact, cfg)
            if m.state == "AMBIGUOUS":
                ambiguous(i, e, f"Airborne fielder {e.fielder_id}'s take-off point is too close to the boundary to call", "Law 19.5.2", [m])
                break
            if first and m.state == "BEYOND":
                beyond(i, e, f"Fielder {e.fielder_id}'s first touch was airborne after last grounding beyond the boundary",
                       "Law 19.5.2", [m])
                break
            if m.state == "BEYOND":
                jumped_outside.add(e.fielder_id)
                item(i, e, f"Fielder {e.fielder_id} jumped from outside the boundary to touch the ball again (allowed once; "
                           "every later ground contact must be inside)", "Law 19.5.2.1", [m])
            else:
                item(i, e, f"Fielder {e.fielder_id} touched the ball while airborne, having taken off inside", "Law 19.5.2", [m])
        elif isinstance(e, FielderGroundContact):
            if e.position is None:
                if e.fielder_id in jumped_outside or e.in_contact_with_ball:
                    ambiguous(i, e, f"Landing position of fielder {e.fielder_id} was not recorded", "Law 19.5.2.1")
                    break
                item(i, e, f"Fielder {e.fielder_id} landed (position not recorded)")
                continue
            m = _measure(rope, e.position, cfg)
            relevant = e.fielder_id in jumped_outside or e.in_contact_with_ball is True
            if relevant and m.state == "BEYOND":
                law = "Law 19.4.2" if e.in_contact_with_ball else "Law 19.5.2.1 / 19.5.2.2"
                why = "while in contact with the ball" if e.in_contact_with_ball else "after jumping from outside to touch the ball"
                beyond(i, e, f"Fielder {e.fielder_id} grounded beyond the boundary {why}", law, [m])
                break
            if relevant and m.state == "AMBIGUOUS":
                ambiguous(i, e, f"Fielder {e.fielder_id}'s landing is too close to the boundary to call", "Law 19.5", [m])
                break
            if e.in_contact_with_ball is None and m.state != "INSIDE" and e.fielder_id in touched:
                ambiguous(i, e, f"Fielder {e.fielder_id} landed beyond; whether still in contact with the ball is unknown", "Law 19.4.2", [m])
                break
            item(i, e, f"Fielder {e.fielder_id} ground contact {'inside' if m.state == 'INSIDE' else 'beyond'} the boundary", "Law 19.5", [m])
        elif isinstance(e, FielderRopeContact):
            if e.in_contact_with_ball or e.fielder_id in jumped_outside:
                beyond(i, e, f"Fielder {e.fielder_id} touched the boundary "
                             f"{'while in contact with the ball' if e.in_contact_with_ball else 'after jumping from outside'}",
                       "Law 19.4.2 / 19.5.2.1")
                break
            if e.in_contact_with_ball is None and e.fielder_id in touched:
                ambiguous(i, e, f"Fielder {e.fielder_id} touched the boundary; whether in contact with the ball is unknown", "Law 19.4.2")
                break
            item(i, e, f"Fielder {e.fielder_id} touched the boundary while not in contact with the ball", "Law 19.5.1")
        elif isinstance(e, BallCrossedRope):
            item(i, e, "Ball crossed the boundary line" + (" in the air (not a boundary by itself)" if e.in_air else ""), "Law 33.2.2.4")
        elif isinstance(e, CatchCompleted):
            sim.catch_at = i
            item(i, e, f"Catch completed by fielder {e.fielder_id} (control of ball and own movement); the ball is dead",
                 "Law 33.3 / 20.1.1.3", decisive=stop_at_catch)
            break
        elif isinstance(e, BallDead):
            sim.dead_at = i
            item(i, e, "Ball dead")
            break
    if any(e.source == "model" for e in ev):
        sim.warnings.append("Some events come from an unvalidated model; check them on the frames")
    return sim


class BoundaryResult(BaseModel):
    engine_version: str = ENGINE_VERSION
    status: BoundaryOutcome
    boundary_scored: bool | None
    runs_allowance: int | None = None
    rule_checks: list[RuleCheck] = Field(default_factory=list)
    timeline: list[TimelineItem] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rules_source: str = RULES_SOURCE
    disclaimer: str = DISCLAIMER


def analyze_boundary(events: list[BoundaryEvent], rope: BoundaryRope, facts: BoundaryFacts | None = None,
                     cfg: BoundaryConfig | None = None) -> BoundaryResult:
    facts, cfg = facts or BoundaryFacts(), cfg or BoundaryConfig()
    if not events:
        return BoundaryResult(status=BoundaryOutcome.INSUFFICIENT_DATA, boundary_scored=None, reasons=["No events recorded"])
    sim = _simulate(events, rope, cfg, stop_at_catch=False)
    warnings = sim.warnings + ["Outcome assumes every relevant contact was recorded; unobserved contacts cannot be excluded"]
    res = BoundaryResult(status=BoundaryOutcome.INCONCLUSIVE, boundary_scored=None, timeline=sim.timeline,
                         reasons=list(sim.reasons), warnings=warnings)
    if sim.ambiguous_at is not None:
        return res
    if sim.grounded_beyond_at is None:
        if sim.dead_at is None and sim.catch_at is None:
            res.reasons.append("Sequence ends before the ball is dead; a later boundary cannot be excluded")
            return res
        res.status, res.boundary_scored = BoundaryOutcome.NO_BOUNDARY, False
        res.reasons.append("Neither the ball nor a fielder in contact with it was grounded beyond the boundary before the ball was dead")
        if sim.catch_at is not None:
            res.warnings.append("Sequence ends at the completed catch; later events were not considered")
        return res
    res.boundary_scored = True
    six_check = RuleCheck(id="boundary_six", description="Struck by the bat and first grounded beyond without touching the ground inside",
                          law_reference="Law 19.7.1", status=CheckStatus.UNKNOWN, evidence="")
    if sim.grounded_inside_before:
        six_check.status, six_check.evidence = CheckStatus.FAIL, "Ball touched the ground inside the field first"
        res.status = BoundaryOutcome.BOUNDARY_FOUR
    elif facts.struck_by_bat is None:
        six_check.evidence = "Whether the ball was struck by the bat was not supplied"
        res.reasons.append("Boundary scored, but 6 or 4 depends on whether the ball was struck by the bat")
    elif facts.struck_by_bat:
        six_check.status, six_check.evidence = CheckStatus.PASS, "Struck by the bat (supplied); no recorded ground contact inside"
        res.status = BoundaryOutcome.BOUNDARY_SIX
        res.warnings.append("A 6 relies on no unrecorded ground contact inside the field before the ball went beyond")
    else:
        six_check.status, six_check.evidence = CheckStatus.FAIL, "Not struck by the bat (supplied)"
        res.status = BoundaryOutcome.BOUNDARY_FOUR
    res.rule_checks.append(six_check)
    if res.status == BoundaryOutcome.BOUNDARY_SIX:
        res.runs_allowance = cfg.boundary_6_allowance
    elif res.status == BoundaryOutcome.BOUNDARY_FOUR:
        res.runs_allowance = cfg.boundary_4_allowance
    return res


class CatchResult(BaseModel):
    engine_version: str = ENGINE_VERSION
    status: CatchOutcome
    rule_checks: list[RuleCheck] = Field(default_factory=list)
    timeline: list[TimelineItem] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    scope: Literal["fielding_and_boundary_conditions"] = "fielding_and_boundary_conditions"
    rules_source: str = RULES_SOURCE
    disclaimer: str = DISCLAIMER


def analyze_catch_near_boundary(events: list[BoundaryEvent], rope: BoundaryRope, facts: BoundaryFacts | None = None,
                                cfg: BoundaryConfig | None = None) -> CatchResult:
    """Was the catch fair with respect to the ground and the boundary (Laws 33.1-33.3, 19.4, 19.5)?

    ``status`` covers these fielding conditions only; No ball and bat contact are reported as rule checks.
    """
    facts, cfg = facts or BoundaryFacts(), cfg or BoundaryConfig()
    if not events:
        return CatchResult(status=CatchOutcome.INSUFFICIENT_DATA, reasons=["No events recorded"])
    sim = _simulate(events, rope, cfg, stop_at_catch=True)
    res = CatchResult(status=CatchOutcome.INCONCLUSIVE, timeline=sim.timeline, reasons=list(sim.reasons),
                      warnings=sim.warnings + ["Assumes every contact up to the completed catch was recorded"])
    checks = res.rule_checks
    checks.append(RuleCheck(id="fair_delivery", description="Delivery is not a No ball", law_reference="Law 33.1",
                            status=CheckStatus.UNKNOWN if facts.no_ball is None else (CheckStatus.PASS if not facts.no_ball else CheckStatus.FAIL),
                            evidence="User-supplied fact" if facts.no_ball is not None else "Not determined by this system"))
    checks.append(RuleCheck(id="touched_bat", description="Ball touched the striker's bat", law_reference="Law 33.1",
                            status=CheckStatus.UNKNOWN if facts.ball_touched_bat is None else (CheckStatus.PASS if facts.ball_touched_bat else CheckStatus.FAIL),
                            evidence="User-supplied / contact-analysis fact" if facts.ball_touched_bat is not None else "Bat contact not established"))
    if sim.ambiguous_at is not None:
        return res
    if sim.grounded_beyond_at is not None:
        res.status = CatchOutcome.CATCH_INVALID
        res.reasons.append("Ball or a fielder in contact with it was grounded beyond the boundary before the catch was completed")
        checks.append(RuleCheck(id="not_grounded_beyond", description="Neither ball nor fielder in contact with it grounded beyond",
                                law_reference="Law 33.2.1 / 19.4 / 19.5", status=CheckStatus.FAIL, evidence=sim.reasons[-1]))
        return res
    if sim.ground_inside_at is not None:
        res.status = CatchOutcome.CATCH_INVALID
        res.reasons.append("Ball touched the ground before the catch was completed")
        checks.append(RuleCheck(id="held_before_ground", description="Ball held before it touched the ground",
                                law_reference="Law 33.1", status=CheckStatus.FAIL, evidence=res.reasons[-1]))
        return res
    if sim.catch_at is None:
        res.reasons.append("No completed catch recorded (control of ball and own movement, Law 33.3)")
        return res
    res.status = CatchOutcome.CATCH_VALID
    res.reasons.append("Catch completed with no recorded ground contact by the ball and no grounding beyond the boundary")
    checks.append(RuleCheck(id="not_grounded_beyond", description="Neither ball nor fielder in contact with it grounded beyond",
                            law_reference="Law 33.2.1 / 19.4 / 19.5", status=CheckStatus.PASS, evidence="No such event before the catch"))
    checks.append(RuleCheck(id="held_before_ground", description="Ball held before it touched the ground",
                            law_reference="Law 33.1", status=CheckStatus.PASS, evidence="No recorded ball-ground contact"))
    return res
