"""No-ball (bowler's foot) analysis: geometry from ml.runout.crease, Law 21.5 rules here.

Laws of Cricket (MCC, 2017 Code 4th Edition, 2026), Law 21.5 *Fair delivery - the feet*: for a
delivery to be fair in respect of the feet, in the delivery stride
* 21.5.1 the bowler's back foot must land within and not touching the return crease appertaining
  to the stated mode of delivery;
* 21.5.2 the bowler's front foot must land with some part of the foot, whether grounded or raised,
  on the same side of the imaginary line joining the two middle stumps as the return crease
  described in 21.5.1, and *behind the popping crease*.
Law 7.3: the popping crease is the back edge of the crease marking, so a foot entirely on the
marking is not behind it.

Measurements are of operator-marked image points (or model proposals the operator accepted):
* front foot: the *rearmost* part of the front foot at landing, marked at ground level beneath it
  (a raised heel counts, Law 21.5.2, so mark where it is above the ground, not the raised point);
* back foot: the part of the back foot *nearest the return crease* at landing.
Uncertainty works as for run-outs: a point within the localisation uncertainty of a line is
AMBIGUOUS and the result is INCONCLUSIVE, never a guess. The middle-stump-line side condition of
21.5.2 is taken as an operator fact (it concerns the stated mode of delivery, over/round the wicket).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ml.lbw.rules import CheckStatus, RuleCheck
from ml.runout.crease import CreaseLine, CreaseMeasurement, measure_point

ENGINE_VERSION = "noball-0.1.0"
RULES_SOURCE = "MCC Laws of Cricket, 2017 Code 4th Edition (2026), Law 21.5 and Law 7.3"
DISCLAIMER = ("Experimental analysis of operator-marked foot positions. Not an official no-ball call; "
              "the umpire decides.")


class FootMark(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_ms: float
    point: tuple[float, float] = Field(description="image xy at ground level (see module docstring for which part)")
    sigma_px: float | None = Field(None, gt=0)
    source: Literal["manual", "model"] = "manual"


class NoBallFacts(BaseModel):
    front_foot_correct_side_of_middle_stump_line: bool | None = Field(
        None, description="Law 21.5.2: same side of the middle-stump line as the return crease of the stated mode")
    landing_frame_is_first_contact: bool | None = Field(
        None, description="operator confirms the marked front-foot frame is the first frame the foot is down")


class NoBallResult(BaseModel):
    engine_version: str = ENGINE_VERSION
    status: Literal["FAIR", "NO_BALL", "INCONCLUSIVE", "INSUFFICIENT_DATA"]
    front_foot: CreaseMeasurement | None = None
    back_foot: CreaseMeasurement | None = None
    front_foot_frame: int | None = None
    back_foot_frame: int | None = None
    rule_checks: list[RuleCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rules_source: str = RULES_SOURCE
    disclaimer: str = DISCLAIMER


def _state_to_status(state: str) -> CheckStatus:
    return {"BEHIND": CheckStatus.PASS, "NOT_BEHIND": CheckStatus.FAIL}.get(state, CheckStatus.UNKNOWN)


def analyze_no_ball(
    popping_crease: CreaseLine | None,
    front_marks: list[FootMark],
    return_crease: CreaseLine | None = None,
    back_marks: list[FootMark] | None = None,
    facts: NoBallFacts | None = None,
    default_sigma_px: float = 2.0,
    decision_confidence: float = 0.95,
) -> NoBallResult:
    """Fair / no ball from the bowler's foot positions at landing.

    ``popping_crease``: the bowler's-end popping crease with ``behind_ref`` on the bowler's stumps side.
    ``return_crease``: the return crease with ``behind_ref`` on the *inside* (towards the middle stump).
    For each foot the earliest mark is taken as the landing frame.
    """
    facts = facts or NoBallFacts()
    res = NoBallResult(status="INSUFFICIENT_DATA")
    checks: list[RuleCheck] = []

    if popping_crease is None or not front_marks:
        res.reasons.append("Front-foot landing mark and the popping crease are both required (Law 21.5.2)")
        res.rule_checks = [RuleCheck(id="front_foot_behind_crease", description="Some part of the front foot behind the popping crease",
                                     law_reference="Law 21.5.2", status=CheckStatus.UNKNOWN, evidence="Not marked")]
        return res
    if popping_crease.end not in (None, "bowler"):
        res.warnings.append("The popping crease given is not marked as the bowler's end")
    front = min(front_marks, key=lambda m: m.timestamp_ms)
    fm = measure_point(popping_crease, front.point, front.sigma_px or default_sigma_px, decision_confidence)
    res.front_foot, res.front_foot_frame = fm, front.frame_index
    dist = f"{fm.signed_m * 100:+.1f} cm" if fm.signed_m is not None else f"{fm.signed_px:+.1f} px"
    checks.append(RuleCheck(
        id="front_foot_behind_crease", description="Some part of the front foot, grounded or raised, behind the popping crease",
        law_reference="Law 21.5.2 / 7.3", status=_state_to_status(fm.state), probability=fm.p_behind,
        evidence=f"Rearmost front-foot point at frame {front.frame_index} is {dist} from the crease's back edge "
                 f"(+ = behind), P(behind) = {fm.p_behind:.3f}, {fm.basis} measurement"))
    side = facts.front_foot_correct_side_of_middle_stump_line
    checks.append(RuleCheck(
        id="front_foot_side", description="Front foot on the same side of the middle-stump line as the return crease of the stated mode",
        law_reference="Law 21.5.2", status=CheckStatus.UNKNOWN if side is None else (CheckStatus.PASS if side else CheckStatus.FAIL),
        evidence="Operator fact" if side is not None else "Not determined (stated mode of delivery)"))

    if return_crease is not None and back_marks:
        back = min(back_marks, key=lambda m: m.timestamp_ms)
        bm = measure_point(return_crease, back.point, back.sigma_px or default_sigma_px, decision_confidence)
        res.back_foot, res.back_foot_frame = bm, back.frame_index
        bdist = f"{bm.signed_m * 100:+.1f} cm" if bm.signed_m is not None else f"{bm.signed_px:+.1f} px"
        checks.append(RuleCheck(
            id="back_foot_within_return_crease", description="Back foot lands within and not touching the return crease",
            law_reference="Law 21.5.1", status=_state_to_status(bm.state), probability=bm.p_behind,
            evidence=f"Back-foot point nearest the return crease at frame {back.frame_index} is {bdist} inside its inner edge "
                     f"(+ = inside), P(inside) = {bm.p_behind:.3f}"))
    else:
        checks.append(RuleCheck(id="back_foot_within_return_crease", description="Back foot lands within and not touching the return crease",
                                law_reference="Law 21.5.1", status=CheckStatus.UNKNOWN, evidence="Back foot or return crease not marked"))

    if facts.landing_frame_is_first_contact is not True:
        res.warnings.append("Confirm the marked front-foot frame is the first frame the foot is down: the foot can slide after landing")
    gaps = sorted({abs(a.timestamp_ms - b.timestamp_ms) for a, b in zip(front_marks, front_marks[1:]) if a.timestamp_ms != b.timestamp_ms})
    if gaps and gaps[0] > 10:
        res.warnings.append(f"Frames are {gaps[0]:.0f} ms apart: the exact landing instant lies between frames")
    res.warnings += popping_crease.warnings

    res.rule_checks = checks
    st = {c.id: c.status for c in checks}
    if CheckStatus.FAIL in (st["front_foot_behind_crease"], st["back_foot_within_return_crease"], st["front_foot_side"]):
        res.status = "NO_BALL"
        failed = [c for c in checks if c.status == CheckStatus.FAIL]
        res.reasons += [f"{c.law_reference}: {c.description} - not satisfied ({c.evidence})" for c in failed]
    elif st["front_foot_behind_crease"] == CheckStatus.PASS:
        res.status = "FAIR"
        res.reasons.append("Part of the front foot is behind the popping crease at landing (Law 21.5.2)")
        unknown = [c.law_reference for c in checks if c.status == CheckStatus.UNKNOWN]
        if unknown:
            res.reasons.append(f"Not checked: {', '.join(unknown)}")
    else:
        res.status = "INCONCLUSIVE"
        res.reasons.append("The front foot's rearmost point is within measurement uncertainty of the crease's back edge")
    return res
