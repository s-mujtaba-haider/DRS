"""Configurable LBW rules layer (MCC Law 36 conditions), separate from geometry.

Each condition evaluates to PASS / FAIL / UNKNOWN / NOT_APPLICABLE from
probabilities produced by the geometry layer plus user-supplied facts. A
condition is only PASS or FAIL when the evidence clears the configured
confidence threshold; otherwise it is UNKNOWN. Unknown inputs stay unknown.

This is an analytical aid. It does not reproduce every clause of the Laws or of
any competition's playing conditions and is not an umpiring decision.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

Tri = Literal[True, False, None]


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RuleCheck(BaseModel):
    id: str
    description: str
    law_reference: str
    status: CheckStatus
    probability: float | None = None
    evidence: str


class RulesConfig(BaseModel):
    source: str = "MCC Laws of Cricket, 2017 Code 4th edition (in force 1 Oct 2026), Law 36 Leg before wicket"
    decision_confidence: float = Field(0.95, ge=0.5, lt=1.0)
    # When True, "in line" requires only part of the ball to be between wicket and wicket.
    in_line_any_part_of_ball: bool = True


class LbwFacts(BaseModel):
    """Facts the geometry cannot establish; None = unknown."""

    batter_handedness: Literal["right", "left"] | None = None
    no_ball: bool | None = None
    ball_touched_bat_first: bool | None = None
    genuine_attempt_to_play: bool | None = None
    intercepted_by_person_not_bat_hand: bool | None = None


def _status_from_p(p: float | None, conf: float) -> CheckStatus:
    if p is None or not np.isfinite(p):
        return CheckStatus.UNKNOWN
    if p >= conf:
        return CheckStatus.PASS
    if p <= 1 - conf:
        return CheckStatus.FAIL
    return CheckStatus.UNKNOWN


def _fact(v: bool | None, want: bool) -> CheckStatus:
    if v is None:
        return CheckStatus.UNKNOWN
    return CheckStatus.PASS if v == want else CheckStatus.FAIL


def side_probabilities(x_samples: np.ndarray, half_width: float, ball_radius: float, handed: str | None,
                       any_part: bool) -> dict[str, float | None]:
    """P(in line), P(outside off), P(outside leg) for lateral positions x (pitch frame)."""
    x = x_samples[np.isfinite(x_samples)]
    if len(x) == 0:
        return {"in_line": None, "outside_off": None, "outside_leg": None}
    lim = half_width + (ball_radius if any_part else 0.0)
    in_line = float(np.mean(np.abs(x) <= lim))
    if handed is None:
        return {"in_line": in_line, "outside_off": None, "outside_leg": None}
    off_sign = 1.0 if handed == "right" else -1.0
    off = float(np.mean(off_sign * x > lim))
    leg = float(np.mean(off_sign * x < -lim))
    return {"in_line": in_line, "outside_off": off, "outside_leg": leg}


def evaluate_rules(
    cfg: RulesConfig,
    facts: LbwFacts,
    pitched: bool | None,
    pitch_x_samples: np.ndarray | None,
    impact_x_samples: np.ndarray | None,
    p_hit: float | None,
    half_width: float,
    ball_radius: float,
) -> tuple[list[RuleCheck], str]:
    conf = cfg.decision_confidence
    h = facts.batter_handedness
    checks: list[RuleCheck] = []

    checks.append(RuleCheck(id="fair_delivery", description="Delivery is not a no ball", law_reference="Law 36.1.1",
                            status=_fact(facts.no_ball, False), evidence="User-supplied fact" if facts.no_ball is not None else "Not determined by this system"))
    checks.append(RuleCheck(id="interception", description="Ball intercepted by part of the striker's person (not bat or hand holding bat)",
                            law_reference="Law 36.1.3", status=_fact(facts.intercepted_by_person_not_bat_hand, True),
                            evidence="User-supplied fact" if facts.intercepted_by_person_not_bat_hand is not None else "Interception body part not classified"))
    checks.append(RuleCheck(id="not_bat_first", description="Ball did not touch the bat before interception",
                            law_reference="Law 36.1.3 / 36.2.2", status=_fact(facts.ball_touched_bat_first, False),
                            evidence="User-supplied / contact-analysis fact" if facts.ball_touched_bat_first is not None else "Bat contact not established"))

    # Pitching: full tosses are exempt; otherwise must not pitch outside leg.
    if pitched is False:
        checks.append(RuleCheck(id="pitching", description="Ball did not pitch outside leg stump",
                                law_reference="Law 36.1.2", status=CheckStatus.NOT_APPLICABLE,
                                evidence="Model indicates the ball was intercepted full pitch"))
    elif pitched is None or pitch_x_samples is None:
        checks.append(RuleCheck(id="pitching", description="Ball did not pitch outside leg stump",
                                law_reference="Law 36.1.2", status=CheckStatus.UNKNOWN, evidence="Pitching point not established"))
    else:
        sp = side_probabilities(pitch_x_samples, half_width, ball_radius, h, cfg.in_line_any_part_of_ball)
        p_ok = None if sp["outside_leg"] is None else 1.0 - sp["outside_leg"]
        checks.append(RuleCheck(id="pitching", description="Ball did not pitch outside leg stump", law_reference="Law 36.1.2",
                                status=_status_from_p(p_ok, conf), probability=p_ok,
                                evidence="Batter handedness unknown, so leg side is undefined" if h is None else
                                f"P(pitched outside leg) = {sp['outside_leg']:.3f}"))

    # Impact: in line, or outside off when no genuine attempt to play.
    if impact_x_samples is None:
        checks.append(RuleCheck(id="impact", description="Impact in line, or outside off with no genuine attempt to play",
                                law_reference="Law 36.1.4", status=CheckStatus.UNKNOWN, evidence="Impact point not established"))
    else:
        sp = side_probabilities(impact_x_samples, half_width, ball_radius, h, cfg.in_line_any_part_of_ball)
        p_ok = sp["in_line"]
        note = f"P(impact in line) = {sp['in_line']:.3f}"
        if facts.genuine_attempt_to_play is False and sp["outside_off"] is not None:
            p_ok = sp["in_line"] + sp["outside_off"]
            note += f"; no shot offered, so outside off also qualifies (P = {sp['outside_off']:.3f})"
        elif facts.genuine_attempt_to_play is None:
            note += "; whether a shot was offered is unknown"
        status = _status_from_p(p_ok, conf)
        if status == CheckStatus.FAIL and facts.genuine_attempt_to_play is None and sp["outside_off"] not in (None, 0.0):
            status = CheckStatus.UNKNOWN  # could still qualify if no shot was offered
        checks.append(RuleCheck(id="impact", description="Impact in line, or outside off with no genuine attempt to play",
                                law_reference="Law 36.1.4", status=status, probability=p_ok, evidence=note))

    checks.append(RuleCheck(id="would_hit_wicket", description="Ball would have hit the wicket but for the interception",
                            law_reference="Law 36.1.5 / 36.2.3", status=_status_from_p(p_hit, conf), probability=p_hit,
                            evidence="Monte Carlo projection of the fitted flight model" if p_hit is not None else "Projection unavailable"))

    statuses = [c.status for c in checks]
    if CheckStatus.FAIL in statuses:
        summary = "CONDITIONS_NOT_MET"
    elif all(s in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE) for s in statuses):
        summary = "ALL_CONDITIONS_MET"
    else:
        summary = "UNDETERMINED"
    return checks, summary
