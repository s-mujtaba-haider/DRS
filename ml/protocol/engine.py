"""DRS review protocol layer: turns evidence-engine output into a review outcome.

The evidence engines (ml.lbw, run-out, boundary, ...) answer geometric questions with
probabilities. A competition's DRS protocol then decides what a review means: the
umpire's-call bands, which on-field decisions can be overturned, and how many
unsuccessful reviews a team has. Those rules change between competitions and
seasons, so they live in versioned YAML files under configs/drs_protocol/ and are
kept out of the vision and geometry code.

Every component is decided only when one category reaches the protocol's
confidence threshold; otherwise it is UNRESOLVED and the review is INCONCLUSIVE.
Facts the system cannot observe (no ball, bat first, interception by the person)
must be confirmed before a not-out is overturned. Nothing here is an official
decision: an INCONCLUSIVE outcome means a human third umpire decides.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROTOCOL_DIR = Path(__file__).resolve().parents[2] / "configs" / "drs_protocol"
DISCLAIMER = ("Experimental protocol interpretation of experimental evidence. Not an official DRS decision; "
              "the third umpire decides.")

OnField = Literal["OUT", "NOT_OUT"]
FinalDecision = Literal["OUT", "NOT_OUT", "REFER_TO_THIRD_UMPIRE"]


class ReviewOutcome(StrEnum):
    UPHELD = "UPHELD"
    OVERTURNED = "OVERTURNED"
    UMPIRES_CALL = "UMPIRES_CALL"
    INCONCLUSIVE = "INCONCLUSIVE"


class ComponentStatus(StrEnum):
    SUPPORTS_OUT = "SUPPORTS_OUT"
    SUPPORTS_NOT_OUT = "SUPPORTS_NOT_OUT"
    UMPIRES_CALL = "UMPIRES_CALL"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ----------------------------------------------------------------------------- protocol model
class Source(BaseModel):
    title: str
    url: str


class DistanceRule(BaseModel):
    """On-field NOT OUT stands (umpire's call) when the impact is far from the stumps."""

    enabled: bool = False
    impact_distance_m: float = 3.0  # impact at or beyond this distance
    extended_impact_distance_m: float = 2.5  # or beyond this distance ...
    extended_max_pitch_to_impact_m: float = 0.4  # ... with pitch-to-impact distance below this


class ReviewRules(BaseModel):
    player_reviews_allowed: bool = True
    unsuccessful_reviews_per_innings: int | None = Field(None, ge=0, description="None = unlimited")
    top_up_after_overs: int | None = None
    umpires_call_retains_review: bool = True
    request_time_limit_s: float | None = 15
    not_reviewable_by_players: list[str] = Field(default_factory=lambda: ["timed_out"])


class LbwProtocol(BaseModel):
    umpires_call_enabled: bool = True
    wicket_zone_height: Literal["top_of_bails", "top_of_stumps"] = "top_of_bails"
    min_component_confidence: float = Field(0.95, ge=0.5, lt=1.0)
    distance_rule: DistanceRule = Field(default_factory=DistanceRule)


class DrsProtocol(BaseModel):
    id: str
    name: str
    version: str
    format: str = "generic"
    effective_from: str
    sources: list[Source] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)
    reviews: ReviewRules = Field(default_factory=ReviewRules)
    lbw: LbwProtocol = Field(default_factory=LbwProtocol)


@lru_cache
def load_protocol(protocol_id: str) -> DrsProtocol:
    import yaml

    if not protocol_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid protocol id '{protocol_id}'")
    path = PROTOCOL_DIR / f"{protocol_id}.yaml"
    if not path.is_file():
        raise KeyError(f"unknown DRS protocol '{protocol_id}'")
    p = DrsProtocol.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if p.id != protocol_id:
        raise ValueError(f"{path.name}: id '{p.id}' does not match the file name")
    return p


def list_protocols() -> list[dict]:
    out = []
    for f in sorted(PROTOCOL_DIR.glob("*.yaml")):
        p = load_protocol(f.stem)
        out.append({"id": p.id, "name": p.name, "version": p.version, "format": p.format,
                    "effective_from": p.effective_from,
                    "unsuccessful_reviews_per_innings": p.reviews.unsuccessful_reviews_per_innings,
                    "umpires_call_enabled": p.lbw.umpires_call_enabled, "unverified": p.unverified})
    return out


# ----------------------------------------------------------------------------- results
class ComponentResult(BaseModel):
    name: str  # pitching | impact | wicket
    status: ComponentStatus
    umpires_call: bool = False
    probabilities: dict[str, float | None] = Field(default_factory=dict)
    evidence: str


class ProtocolDecision(BaseModel):
    protocol_id: str
    protocol_version: str
    review_type: str
    review_kind: Literal["player", "umpire"] = "player"
    on_field: str
    outcome: ReviewOutcome
    final_decision: str
    review_retained: bool | None = Field(None, description="Player reviews only; None when not decided or not applicable")
    consumes_review: bool = False
    components: list[ComponentResult] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


# ----------------------------------------------------------------------------- LBW components
def _pick(probs: dict[str, float | None], conf: float) -> str | None:
    """Category whose probability reaches conf, else None."""
    for k, v in probs.items():
        if v is not None and v >= conf:
            return k
    return None


def _fmt(probs: dict[str, float | None]) -> str:
    return ", ".join(f"P({k}) = {'n/a' if v is None else f'{v:.3f}'}" for k, v in probs.items())


def _pitching(res: dict, conf: float) -> ComponentResult:
    zp = (res.get("zone_probabilities") or {}).get("pitching")
    check = next((c for c in res.get("rule_checks", []) if c.get("id") == "pitching"), {})
    if zp is None:
        if check.get("status") == "NOT_APPLICABLE":
            return ComponentResult(name="pitching", status=ComponentStatus.NOT_APPLICABLE,
                                   evidence="Full toss: the ball did not pitch before interception")
        return ComponentResult(name="pitching", status=ComponentStatus.UNRESOLVED, evidence="Pitching point not established")
    leg = zp.get("centre_outside_leg")
    probs = {"outside_leg": leg, "not_outside_leg": None if leg is None else 1.0 - leg}
    if leg is None:
        return ComponentResult(name="pitching", status=ComponentStatus.UNRESOLVED, probabilities=probs,
                               evidence="Batter handedness unknown, so the leg side is undefined")
    cat = _pick(probs, conf)
    status = {"outside_leg": ComponentStatus.SUPPORTS_NOT_OUT, "not_outside_leg": ComponentStatus.SUPPORTS_OUT}.get(
        cat, ComponentStatus.UNRESOLVED)
    return ComponentResult(name="pitching", status=status, probabilities=probs,
                           evidence=f"Ball centre at pitching (no umpire's call for pitching): {_fmt(probs)}")


def _impact(protocol: DrsProtocol, on_field: OnField, res: dict, attempt: bool | None, conf: float) -> ComponentResult:
    zi = (res.get("zone_probabilities") or {}).get("impact")
    if zi is None:
        return ComponentResult(name="impact", status=ComponentStatus.UNRESOLVED, evidence="Impact point not established")
    uc = protocol.lbw.umpires_call_enabled
    in_line = zi["centre_in_line"] if uc else zi["any_part_in_line"]
    uc_off, uc_leg = (zi.get("umpires_call_band_off"), zi.get("umpires_call_band_leg")) if uc else (0.0, 0.0)
    out_off, out_leg = zi.get("no_part_outside_off"), zi.get("no_part_outside_leg")
    probs = {"in_line": in_line, "umpires_call_off": uc_off, "umpires_call_leg": uc_leg,
             "outside_off": out_off, "outside_leg": out_leg}
    if None in (out_off, out_leg, uc_off, uc_leg):
        # Side unknown: only the in-line / any-part questions are answerable.
        uc_any = zi["any_part_in_line"] - zi["centre_in_line"] if uc else 0.0
        outside = 1.0 - zi["any_part_in_line"]
        p2 = {"in_line": in_line, "umpires_call": uc_any, "outside": outside}
        if attempt is True:
            cat = _pick(p2, conf)
            status = {"in_line": ComponentStatus.SUPPORTS_OUT, "umpires_call": ComponentStatus.UMPIRES_CALL,
                      "outside": ComponentStatus.SUPPORTS_NOT_OUT}.get(cat, ComponentStatus.UNRESOLVED)
            return ComponentResult(name="impact", status=status, umpires_call=status == ComponentStatus.UMPIRES_CALL,
                                   probabilities=p2, evidence=f"Batter handedness unknown; {_fmt(p2)}")
        return ComponentResult(name="impact", status=ComponentStatus.UNRESOLVED, probabilities=p2,
                               evidence="Batter handedness unknown and no confirmed shot: off/leg side needed")

    def classify(shot: bool) -> ComponentStatus:
        if shot:
            groups = {ComponentStatus.SUPPORTS_OUT: in_line, ComponentStatus.UMPIRES_CALL: uc_off + uc_leg,
                      ComponentStatus.SUPPORTS_NOT_OUT: out_off + out_leg}
        else:
            # No genuine attempt to play: impact outside off also qualifies (Law 36.1.4).
            groups = {ComponentStatus.SUPPORTS_OUT: in_line + uc_off + out_off, ComponentStatus.UMPIRES_CALL: uc_leg,
                      ComponentStatus.SUPPORTS_NOT_OUT: out_leg}
        if not uc:
            groups[ComponentStatus.UMPIRES_CALL] = 0.0
        for st, p in groups.items():
            if p >= conf:
                return st
        return ComponentStatus.UNRESOLVED

    if attempt is None:
        a, b = classify(True), classify(False)
        status = a if a == b else ComponentStatus.UNRESOLVED
        note = "whether a shot was offered is unknown" + ("" if a == b else f" and it matters ({a} with a shot, {b} without)")
    else:
        status = classify(attempt)
        note = "genuine attempt to play" if attempt else "no genuine attempt to play (outside off qualifies)"
    return ComponentResult(name="impact", status=status, umpires_call=status == ComponentStatus.UMPIRES_CALL,
                           probabilities=probs, evidence=f"{'Ball centre' if uc else 'Any part of ball'} vs impact zone; {note}; {_fmt(probs)}")


def _wicket(protocol: DrsProtocol, on_field: OnField, res: dict, conf: float) -> ComponentResult:
    wp = res.get("wicket_projection") or {}
    p_hit = wp.get("p_hit")
    if p_hit is None:
        return ComponentResult(name="wicket", status=ComponentStatus.UNRESOLVED, evidence="No projection to the stumps")
    uc = protocol.lbw.umpires_call_enabled
    by_h = wp.get("p_centre_in_zone_by_height") or {}
    pc = by_h.get(protocol.lbw.wicket_zone_height)
    if pc is None and protocol.lbw.wicket_zone_height == "top_of_bails":
        pc = wp.get("p_centre_in_zone")
    if uc and pc is None:
        return ComponentResult(name="wicket", status=ComponentStatus.UNRESOLVED, probabilities={"hitting_any_part": p_hit},
                               evidence="Centre-in-zone probability unavailable for this protocol's zone height")
    probs = ({"hitting": pc, "umpires_call": max(p_hit - pc, 0.0), "missing": 1.0 - p_hit} if uc
             else {"hitting": p_hit, "umpires_call": 0.0, "missing": 1.0 - p_hit})
    cat = _pick(probs, conf)
    status = {"hitting": ComponentStatus.SUPPORTS_OUT, "umpires_call": ComponentStatus.UMPIRES_CALL,
              "missing": ComponentStatus.SUPPORTS_NOT_OUT}.get(cat, ComponentStatus.UNRESOLVED)
    evidence = f"Projected ball vs wicket zone (to {protocol.lbw.wicket_zone_height.replace('_', ' ')}); {_fmt(probs)}"

    dr = protocol.lbw.distance_rule
    if dr.enabled and on_field == "NOT_OUT" and status == ComponentStatus.SUPPORTS_OUT:
        zp = res.get("zone_probabilities") or {}
        d = zp.get("impact_distance_from_stumps_m")
        pti = zp.get("pitch_to_impact_m")
        if d is None:
            return ComponentResult(name="wicket", status=ComponentStatus.UNRESOLVED, probabilities=probs,
                                   evidence=evidence + "; impact distance unknown, distance rule cannot be applied")
        far = d["p05"] >= dr.impact_distance_m
        ext = pti is not None and d["p05"] > dr.extended_impact_distance_m and pti["p95"] < dr.extended_max_pitch_to_impact_m
        not_far = d["p95"] < dr.impact_distance_m
        # A full toss has no pitch-to-impact distance, so the extended rule cannot apply.
        not_ext = pti is None or d["p95"] <= dr.extended_impact_distance_m or pti["p05"] >= dr.extended_max_pitch_to_impact_m
        dist_note = f"; impact {d['mean']:.2f} m from stumps (90% {d['p05']:.2f}-{d['p95']:.2f})"
        if far or ext:
            return ComponentResult(name="wicket", status=ComponentStatus.UMPIRES_CALL, umpires_call=True, probabilities=probs,
                                   evidence=evidence + dist_note + "; distance rule keeps the on-field NOT OUT")
        if not (not_far and not_ext):
            return ComponentResult(name="wicket", status=ComponentStatus.UNRESOLVED, probabilities=probs,
                                   evidence=evidence + dist_note + "; uncertainty straddles a distance-rule threshold")
        evidence += dist_note
    return ComponentResult(name="wicket", status=status, umpires_call=status == ComponentStatus.UMPIRES_CALL,
                           probabilities=probs, evidence=evidence)


_FACT_IDS = {"fair_delivery": "no ball", "not_bat_first": "bat contact before pad",
             "interception": "interception by the striker's person"}


def _player_retained(protocol: DrsProtocol, outcome: ReviewOutcome) -> bool | None:
    if outcome == ReviewOutcome.OVERTURNED:
        return True
    if outcome == ReviewOutcome.UMPIRES_CALL:
        return protocol.reviews.umpires_call_retains_review
    if outcome == ReviewOutcome.UPHELD:
        return False
    return None


def lbw_review(protocol: DrsProtocol, on_field: OnField, lbw_result: dict, review_kind: Literal["player", "umpire"] = "player",
               genuine_attempt_to_play: bool | None = None) -> ProtocolDecision:
    """Apply the protocol to an LbwResult (as dict, e.g. LbwResult.model_dump(mode='json')).

    genuine_attempt_to_play defaults to the facts recorded in the result's rule checks when not given.
    """
    if on_field not in ("OUT", "NOT_OUT"):
        raise ValueError("on_field must be 'OUT' or 'NOT_OUT' for an LBW review")
    conf = protocol.lbw.min_component_confidence
    reasons: list[str] = []
    warnings: list[str] = []
    checks = {c.get("id"): c for c in lbw_result.get("rule_checks", [])}
    if genuine_attempt_to_play is None:
        ev = (checks.get("impact") or {}).get("evidence", "")
        if "no shot offered" in ev:
            genuine_attempt_to_play = False

    comps = [_pitching(lbw_result, conf), _impact(protocol, on_field, lbw_result, genuine_attempt_to_play, conf),
             _wicket(protocol, on_field, lbw_result, conf)]
    fact_fail = [n for k, n in _FACT_IDS.items() if (checks.get(k) or {}).get("status") == "FAIL"]
    fact_unknown = [n for k, n in _FACT_IDS.items() if (checks.get(k) or {}).get("status", "UNKNOWN") == "UNKNOWN"]

    def decide() -> tuple[ReviewOutcome, str]:
        if fact_fail:
            reasons.append("Recorded facts rule out LBW: " + ", ".join(fact_fail))
            return (ReviewOutcome.OVERTURNED, "NOT_OUT") if on_field == "OUT" else (ReviewOutcome.UPHELD, "NOT_OUT")
        not_out = [c.name for c in comps if c.status == ComponentStatus.SUPPORTS_NOT_OUT]
        unresolved = [c.name for c in comps if c.status == ComponentStatus.UNRESOLVED]
        ucs = [c.name for c in comps if c.status == ComponentStatus.UMPIRES_CALL]
        if not_out:
            reasons.append("Not out on: " + ", ".join(not_out))
            return (ReviewOutcome.OVERTURNED, "NOT_OUT") if on_field == "OUT" else (ReviewOutcome.UPHELD, "NOT_OUT")
        if unresolved:
            reasons.append("Unresolved components: " + ", ".join(unresolved))
            return ReviewOutcome.INCONCLUSIVE, on_field
        if ucs:
            reasons.append("Umpire's call on: " + ", ".join(ucs) + "; the on-field decision stands")
            return ReviewOutcome.UMPIRES_CALL, on_field
        if fact_unknown:
            reasons.append("Ball-tracking components support OUT, but these facts are not confirmed: " + ", ".join(fact_unknown))
            warnings.append("The third umpire must confirm no-ball, bat contact and interception before a decision")
            return ReviewOutcome.INCONCLUSIVE, on_field
        reasons.append("All components support OUT and the recorded facts are consistent")
        return (ReviewOutcome.UPHELD, "OUT") if on_field == "OUT" else (ReviewOutcome.OVERTURNED, "OUT")

    outcome, final = decide()
    verdict = lbw_result.get("verdict")
    # An INCONCLUSIVE ball-tracking verdict concerns whether the ball would hit the wicket. It cannot
    # overrule a NOT OUT that rests on a resolved pitching or impact component (each cleared its own
    # confidence bar) or on a recorded fact. INSUFFICIENT_DATA means the track itself is unusable, so
    # nothing derived from it is trusted.
    independent_not_out = bool(fact_fail) or any(
        c.name in ("pitching", "impact") and c.status == ComponentStatus.SUPPORTS_NOT_OUT for c in comps)
    blocked = verdict == "INSUFFICIENT_DATA" or (verdict == "INCONCLUSIVE" and not (final == "NOT_OUT" and independent_not_out))
    if blocked and outcome != ReviewOutcome.INCONCLUSIVE:
        reasons.append(f"Protocol reading would be {outcome.value} ({final}), but the ball-tracking verdict is {verdict}")
        outcome, final = ReviewOutcome.INCONCLUSIVE, on_field
    if outcome == ReviewOutcome.INCONCLUSIVE:
        warnings.append("System evidence is inconclusive: the on-field decision is shown unchanged and the third umpire must decide")
    if not lbw_result.get("calibration_validated", False):
        warnings.append("Camera calibration is not validated")
    if protocol.unverified:
        warnings.append(f"Protocol '{protocol.id}' has unverified items; see its config")
    player = review_kind == "player"
    return ProtocolDecision(
        protocol_id=protocol.id, protocol_version=protocol.version, review_type="lbw", review_kind=review_kind,
        on_field=on_field, outcome=outcome, final_decision=final,
        review_retained=_player_retained(protocol, outcome) if player else None,
        consumes_review=player and outcome == ReviewOutcome.UPHELD,
        components=comps, reasons=reasons, warnings=warnings)


# ----------------------------------------------------------------------------- other reviews
EVIDENCE_TO_DECISION = {
    "OUT": "OUT", "OUT_OF_GROUND": "OUT", "CAUGHT": "OUT", "CLEAN_CATCH": "OUT", "STUMPED": "OUT",
    "NOT_OUT": "NOT_OUT", "IN_GROUND": "NOT_OUT", "NOT_CAUGHT": "NOT_OUT", "BALL_GROUNDED": "NOT_OUT",
    "NOT_STUMPED": "NOT_OUT",
    "SIX": "SIX", "FOUR": "FOUR", "NO_BOUNDARY": "NO_BOUNDARY",
}


def simple_review(protocol: DrsProtocol, on_field: str, evidence_status: str, review_type: str = "run_out",
                  review_kind: Literal["player", "umpire"] = "player") -> ProtocolDecision:
    """Map an evidence-engine status onto a review outcome for non-LBW decisions.

    on_field: OUT / NOT_OUT (dismissals), FOUR / SIX / NO_BOUNDARY (boundary), or NONE for an umpire
    referral with no on-field call (ICC removed the soft signal from 1 Jun 2023).
    Umpire referrals never consume a team's review.
    """
    on_field = on_field.upper()
    ev = evidence_status.upper()
    reasons: list[str] = []
    warnings: list[str] = []
    if review_kind == "player":
        if not protocol.reviews.player_reviews_allowed:
            raise ValueError(f"protocol '{protocol.id}' does not allow player reviews")
        if review_type in protocol.reviews.not_reviewable_by_players:
            raise ValueError(f"'{review_type}' decisions cannot be reviewed by players under '{protocol.id}'")
        if on_field not in ("OUT", "NOT_OUT"):
            raise ValueError("a player review needs an on-field OUT or NOT_OUT decision")
    decision = EVIDENCE_TO_DECISION.get(ev)
    if decision is None:
        outcome = ReviewOutcome.INCONCLUSIVE
        final = on_field if on_field != "NONE" else "REFER_TO_THIRD_UMPIRE"
        reasons.append(f"Evidence status '{evidence_status}' does not settle the decision")
        warnings.append("System evidence is inconclusive: the third umpire must decide")
    elif on_field == "NONE":
        # Nothing to uphold or overturn: the outcome is reported as UPHELD and the decision comes from the evidence.
        outcome, final = ReviewOutcome.UPHELD, decision
        reasons.append(f"No on-field call; evidence indicates {decision}")
    elif decision == on_field:
        outcome, final = ReviewOutcome.UPHELD, on_field
        reasons.append(f"Evidence ({evidence_status}) agrees with the on-field {on_field}")
    else:
        outcome, final = ReviewOutcome.OVERTURNED, decision
        reasons.append(f"Evidence ({evidence_status}) contradicts the on-field {on_field}")
    player = review_kind == "player"
    return ProtocolDecision(
        protocol_id=protocol.id, protocol_version=protocol.version, review_type=review_type, review_kind=review_kind,
        on_field=on_field, outcome=outcome, final_decision=final,
        review_retained=_player_retained(protocol, outcome) if player else None,
        consumes_review=player and outcome == ReviewOutcome.UPHELD, reasons=reasons, warnings=warnings)


# ----------------------------------------------------------------------------- review allowance
def review_counts(protocol: DrsProtocol, reviews: list[dict]) -> dict[str, dict]:
    """Per-team review usage for each team's latest innings.

    Each review: {team, innings, kind: "player"|"umpire", outcome}. Outcome UPHELD is unsuccessful;
    UMPIRES_CALL is retained when the protocol says so (otherwise unsuccessful); OVERTURNED is
    successful; INCONCLUSIVE or missing outcomes are pending and not counted.
    """
    limit = protocol.reviews.unsuccessful_reviews_per_innings
    per: dict[tuple[str, int], dict] = {}
    for r in reviews:
        if r.get("kind", "player") != "player":
            continue
        key = (str(r["team"]), int(r.get("innings", 1)))
        d = per.setdefault(key, {"unsuccessful_used": 0, "successful": 0, "retained_umpires_call": 0, "pending": 0})
        o = str(r.get("outcome") or "").upper()
        if o == ReviewOutcome.UPHELD:
            d["unsuccessful_used"] += 1
        elif o == ReviewOutcome.UMPIRES_CALL:
            if protocol.reviews.umpires_call_retains_review:
                d["retained_umpires_call"] += 1
            else:
                d["unsuccessful_used"] += 1
        elif o == ReviewOutcome.OVERTURNED:
            d["successful"] += 1
        else:
            d["pending"] += 1
    out: dict[str, dict] = {}
    for (team, inn), d in sorted(per.items(), key=lambda kv: kv[0][1]):
        rem = None if limit is None else max(limit - d["unsuccessful_used"], 0)
        entry = {**d, "innings": inn, "limit": limit, "remaining": rem}
        out.setdefault(team, {"by_innings": {}})
        out[team]["by_innings"][inn] = entry
        out[team].update({k: v for k, v in entry.items()})
    return out
