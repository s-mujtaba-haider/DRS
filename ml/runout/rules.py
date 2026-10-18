"""Run-out (Law 38) and Stumped (Law 39) rule checks, separate from the crease geometry.

Source: MCC Laws of Cricket, 2017 Code 4th Edition (2026), https://www.lords.org/mcc/the-laws-of-cricket
* Law 29.1 / 29.2: the wicket is broken when a bail is completely removed or a stump struck out;
  fairly when by the ball, or by a fielder's hand or arm with the ball held in it (29.2.1.5).
* Law 38.1: out Run out if out of ground and the wicket is fairly broken by the action of a fielder
  or the ball rebounding off a fielder, whether or not a run is attempted, even off a No ball.
* Law 38.2.1.1: not out if the batter had been in their ground and left it to avoid injury.
* Law 38.2.1.2: not out if the delivered ball has not touched a fielder before the wicket is broken.
* Law 38.2.2.2: the striker is not Run out if No ball is called, they are out of ground not attempting
  a run, and the wicket-keeper breaks the wicket without the intervention of another fielder.
* Law 39.1.1: out Stumped if the ball is not a No ball, the striker is out of ground (other than
  39.3.1) and has not attempted a run, when the wicket-keeper fairly breaks the wicket without the
  intervention of another fielder. 39.2: rebounds off / kicked or thrown by the keeper count.
* Law 39.3.1: not out Stumped if the striker left their ground to avoid injury.

Each condition is PASS / FAIL / UNKNOWN / NOT_APPLICABLE. Facts the video cannot establish
stay UNKNOWN unless supplied. This is an analytical aid, not an umpiring decision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ml.lbw.rules import CheckStatus, RuleCheck
from ml.runout.crease import CreaseLine
from ml.runout.engine import (
    DISCLAIMER,
    BatterObservation,
    CreaseConfig,
    CreaseEventResult,
    CreaseFacts,
    CreaseStatus,
    TimelineEvent,
    WicketBreak,
    analyze_crease_event,
)

RULES_SOURCE = "MCC Laws of Cricket, 2017 Code 4th Edition (2026): Laws 29, 30, 38, 39"
ENGINE_VERSION = "runout-stumping-0.1.0-experimental"


class DismissalFacts(BaseModel):
    """Facts outside the crease geometry; None = unknown."""

    batter: Literal["striker", "non_striker"] | None = None
    no_ball: bool | None = None
    wicket_broken_fairly: bool | None = Field(None, description="Law 29.2: by the ball, or fielder's hand/arm holding the ball")
    ball_touched_fielder_before_break: bool | None = Field(None, description="Law 38.2.1.2")
    left_ground_to_avoid_injury: bool | None = Field(None, description="Laws 38.2.1.1 / 39.3.1")
    attempting_run: bool | None = None
    broken_by_wicket_keeper_without_intervention: bool | None = Field(None, description="Laws 38.2.2.2 / 39.1.1 / 39.2")
    running_or_diving_into_ground: bool | None = Field(None, description="Law 30.1.2")


class DismissalAnalysis(BaseModel):
    engine_version: str = ENGINE_VERSION
    dismissal: Literal["run_out", "stumped"]
    status: CreaseStatus = Field(description="crease status at the moment the wicket was broken")
    rules_summary: str = Field(description="ALL_CONDITIONS_MET | CONDITIONS_NOT_MET | UNDETERMINED")
    rule_checks: list[RuleCheck]
    crease: CreaseEventResult
    timeline: list[TimelineEvent]
    reasons: list[str]
    warnings: list[str]
    rules_source: str = RULES_SOURCE
    disclaimer: str = DISCLAIMER


def _fact(v: bool | None, want: bool) -> CheckStatus:
    if v is None:
        return CheckStatus.UNKNOWN
    return CheckStatus.PASS if v == want else CheckStatus.FAIL


def _ev(v: bool | None, what: str) -> str:
    return f"User-supplied fact: {what} = {v}" if v is not None else f"Not determined by this system ({what})"


def _common(crease: CreaseEventResult, wb: WicketBreak | None, f: DismissalFacts) -> list[RuleCheck]:
    if wb is None:
        down = RuleCheck(id="wicket_broken", description="Wicket broken: a bail completely removed or a stump removed",
                         law_reference="Law 29.1", status=CheckStatus.UNKNOWN, evidence="No wicket-break interval")
    elif wb.confirmed:
        down = RuleCheck(id="wicket_broken", description="Wicket broken: a bail completely removed or a stump removed",
                         law_reference="Law 29.1", status=CheckStatus.PASS,
                         evidence=f"Operator-confirmed between frames {wb.last_intact_frame} and {wb.first_broken_frame}")
    else:
        down = RuleCheck(id="wicket_broken", description="Wicket broken: a bail completely removed or a stump removed",
                         law_reference="Law 29.1 / 29.2.2", status=CheckStatus.UNKNOWN,
                         evidence=f"{wb.source} candidate between frames {wb.last_intact_frame} and {wb.first_broken_frame}; "
                                  "complete removal not confirmed (a disturbed bail is not removal)")
    fair = RuleCheck(id="broken_fairly", description="Wicket broken fairly (by the ball, or by a fielder's hand/arm holding the ball)",
                     law_reference="Law 29.2", status=_fact(f.wicket_broken_fairly, True), evidence=_ev(f.wicket_broken_fairly, "broken fairly"))
    og = {CreaseStatus.OUT_OF_GROUND: CheckStatus.PASS, CreaseStatus.IN_GROUND: CheckStatus.FAIL}.get(crease.status, CheckStatus.UNKNOWN)
    ground = RuleCheck(id="out_of_ground", description="Batter out of their ground when the wicket was broken",
                       law_reference="Law 30.1", status=og, evidence=f"Crease engine: {crease.status}. " + " ".join(crease.reasons))
    return [down, fair, ground]


def _summary(checks: list[RuleCheck]) -> str:
    st = [c.status for c in checks]
    if CheckStatus.FAIL in st:
        return "CONDITIONS_NOT_MET"
    if all(s in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE) for s in st):
        return "ALL_CONDITIONS_MET"
    return "UNDETERMINED"


def analyze_runout(
    line: CreaseLine,
    observations: list[BatterObservation],
    wicket_break: WicketBreak | None,
    facts: DismissalFacts | None = None,
    cfg: CreaseConfig | None = None,
) -> DismissalAnalysis:
    f = facts or DismissalFacts()
    crease = analyze_crease_event(line, observations, wicket_break, CreaseFacts(running_or_diving_into_ground=f.running_or_diving_into_ground), cfg)
    checks = _common(crease, wicket_break, f)
    checks.append(RuleCheck(id="fielder_contact", description="Delivered ball touched a fielder before the wicket was broken",
                            law_reference="Law 38.2.1.2", status=_fact(f.ball_touched_fielder_before_break, True),
                            evidence=_ev(f.ball_touched_fielder_before_break, "ball touched a fielder")))
    checks.append(RuleCheck(id="not_avoiding_injury", description="Batter did not leave their ground to avoid injury",
                            law_reference="Law 38.2.1.1", status=_fact(f.left_ground_to_avoid_injury, False),
                            evidence=_ev(f.left_ground_to_avoid_injury, "left ground to avoid injury")))
    # 38.2.2.2: striker off a No ball, not attempting a run, keeper breaks without intervention -> not run out
    if f.batter == "non_striker" or f.no_ball is False:
        st, ev = CheckStatus.NOT_APPLICABLE, "Not the striker off a No ball"
    elif f.batter is None or f.no_ball is None:
        st, ev = CheckStatus.UNKNOWN, "Batter role or No ball status unknown"
    elif f.attempting_run is True or f.broken_by_wicket_keeper_without_intervention is False:
        st, ev = CheckStatus.PASS, "Exception does not apply (run attempted or another fielder intervened)"
    elif f.attempting_run is False and f.broken_by_wicket_keeper_without_intervention is True:
        st, ev = CheckStatus.FAIL, "No ball, striker not attempting a run, wicket-keeper broke the wicket unassisted"
    else:
        st, ev = CheckStatus.UNKNOWN, "Run attempt or wicket-keeper involvement unknown"
    checks.append(RuleCheck(id="no_ball_keeper_exception", description="Not the No-ball wicket-keeper exception",
                            law_reference="Law 38.2.2.2", status=st, evidence=ev))
    if f.batter == "striker" and f.no_ball is False and f.attempting_run is False and f.broken_by_wicket_keeper_without_intervention is True:
        checks.append(RuleCheck(id="stumped_precedence", description="Not a stumping (Stumped takes precedence)",
                                law_reference="Law 38.2.2.1 / 39.1.2", status=CheckStatus.FAIL,
                                evidence="Supplied facts satisfy Law 39.1.1; analyse as Stumped"))
    warnings = list(crease.warnings)
    if f.batter is None:
        warnings.append("Which batter's ground is at this end (Law 30.2 / 38.4) was not supplied")
    return DismissalAnalysis(dismissal="run_out", status=crease.status, rules_summary=_summary(checks), rule_checks=checks,
                             crease=crease, timeline=crease.timeline, reasons=list(crease.reasons), warnings=warnings)


def analyze_stumping(
    line: CreaseLine,
    observations: list[BatterObservation],
    wicket_break: WicketBreak | None,
    facts: DismissalFacts | None = None,
    cfg: CreaseConfig | None = None,
) -> DismissalAnalysis:
    f = facts or DismissalFacts()
    crease = analyze_crease_event(line, observations, wicket_break, CreaseFacts(running_or_diving_into_ground=f.running_or_diving_into_ground), cfg)
    checks = _common(crease, wicket_break, f)
    checks.append(RuleCheck(id="fair_delivery", description="Delivery is not a No ball", law_reference="Law 39.1.1",
                            status=_fact(f.no_ball, False), evidence=_ev(f.no_ball, "no ball")))
    checks.append(RuleCheck(id="not_attempting_run", description="Striker has not attempted a run", law_reference="Law 39.1.1",
                            status=_fact(f.attempting_run, False), evidence=_ev(f.attempting_run, "attempting a run")))
    checks.append(RuleCheck(id="keeper_unassisted", description="Wicket broken by the wicket-keeper without the intervention of another fielder",
                            law_reference="Law 39.1.1 / 39.2", status=_fact(f.broken_by_wicket_keeper_without_intervention, True),
                            evidence=_ev(f.broken_by_wicket_keeper_without_intervention, "keeper unassisted")))
    checks.append(RuleCheck(id="not_avoiding_injury", description="Striker did not leave their ground to avoid injury",
                            law_reference="Law 39.3.1", status=_fact(f.left_ground_to_avoid_injury, False),
                            evidence=_ev(f.left_ground_to_avoid_injury, "left ground to avoid injury")))
    if f.batter == "non_striker":
        checks.append(RuleCheck(id="striker", description="Batter is the striker", law_reference="Law 39.1.1",
                                status=CheckStatus.FAIL, evidence="Only the striker can be Stumped"))
    warnings = list(crease.warnings)
    return DismissalAnalysis(dismissal="stumped", status=crease.status, rules_summary=_summary(checks), rule_checks=checks,
                             crease=crease, timeline=crease.timeline, reasons=list(crease.reasons), warnings=warnings)
