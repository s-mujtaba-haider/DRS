"""DRS protocol layer: umpire's call, distance rule, facts, review allowances (deterministic, no geometry)."""

import numpy as np
import pytest

from ml.lbw.engine import zone_probabilities
from ml.protocol import ComponentStatus, ReviewOutcome, lbw_review, list_protocols, load_protocol, review_counts, simple_review

TEST = load_protocol("icc_test_2025")
CLUB = load_protocol("club_no_umpires_call")
FACTS_OK = [{"id": "fair_delivery", "status": "PASS"}, {"id": "not_bat_first", "status": "PASS"},
            {"id": "interception", "status": "PASS"}, {"id": "pitching", "status": "PASS"}, {"id": "impact", "status": "PASS"}]


def result(*, leg=0.0, centre_in=1.0, any_in=1.0, uc_off=0.0, uc_leg=0.0, out_off=0.0, out_leg=0.0, p_hit=1.0, pc=1.0,
           dist=1.5, pti=1.0, verdict="PROJECTED_TO_HIT", checks=FACTS_OK, pitched=True):
    return {
        "verdict": verdict, "calibration_validated": True, "rule_checks": checks,
        "wicket_projection": {"p_hit": p_hit, "p_centre_in_zone": pc, "p_centre_in_zone_by_height": {"top_of_bails": pc, "top_of_stumps": pc}},
        "zone_probabilities": {
            "pitching": {"centre_in_line": 1 - leg, "centre_outside_off": 0.0, "centre_outside_leg": leg} if pitched else None,
            "impact": {"centre_in_line": centre_in, "any_part_in_line": any_in, "umpires_call_band_off": uc_off,
                       "umpires_call_band_leg": uc_leg, "no_part_outside_off": out_off, "no_part_outside_leg": out_leg},
            "impact_distance_from_stumps_m": {"mean": dist, "p05": dist - 0.05, "p95": dist + 0.05},
            "pitch_to_impact_m": {"mean": pti, "p05": pti - 0.05, "p95": pti + 0.05} if pitched else None,
        },
    }


def test_protocols_load_and_are_versioned():
    ids = {p["id"] for p in list_protocols()}
    assert {"icc_test_2025", "icc_odi_2025", "icc_t20i_2025", "club_no_umpires_call"} <= ids
    assert TEST.reviews.unsuccessful_reviews_per_innings == 3
    assert load_protocol("icc_odi_2025").reviews.unsuccessful_reviews_per_innings == 2
    with pytest.raises(KeyError):
        load_protocol("does_not_exist")
    with pytest.raises(ValueError):
        load_protocol("../secrets")


def test_clear_hit_overturns_not_out():
    d = lbw_review(TEST, "NOT_OUT", result())
    assert d.outcome == ReviewOutcome.OVERTURNED and d.final_decision == "OUT" and d.review_retained is True


def test_clear_hit_upholds_out_and_costs_review():
    d = lbw_review(TEST, "OUT", result())
    assert d.outcome == ReviewOutcome.UPHELD and d.final_decision == "OUT"
    assert d.review_retained is False and d.consumes_review


def test_wicket_umpires_call_keeps_decision_and_review():
    r = result(p_hit=1.0, pc=0.0)  # clipping: hitting, centre outside zone
    for on_field in ("OUT", "NOT_OUT"):
        d = lbw_review(TEST, on_field, r)
        assert d.outcome == ReviewOutcome.UMPIRES_CALL and d.final_decision == on_field and d.review_retained is True
        assert next(c for c in d.components if c.name == "wicket").umpires_call


def test_club_protocol_has_no_umpires_call():
    d = lbw_review(CLUB, "NOT_OUT", result(p_hit=1.0, pc=0.0))
    assert d.outcome == ReviewOutcome.OVERTURNED and d.final_decision == "OUT"


def test_missing_overturns_out():
    d = lbw_review(TEST, "OUT", result(p_hit=0.0, pc=0.0, verdict="PROJECTED_TO_MISS"))
    assert d.outcome == ReviewOutcome.OVERTURNED and d.final_decision == "NOT_OUT"


def test_pitching_outside_leg_is_not_out():
    d = lbw_review(TEST, "OUT", result(leg=1.0))
    assert d.outcome == ReviewOutcome.OVERTURNED and d.final_decision == "NOT_OUT"


def test_impact_umpires_call_and_outside_off_without_shot():
    r = result(centre_in=0.0, any_in=1.0, uc_off=1.0)
    assert lbw_review(TEST, "NOT_OUT", r, genuine_attempt_to_play=True).outcome == ReviewOutcome.UMPIRES_CALL
    # No shot offered: impact umpire's call on the off side qualifies for OUT.
    assert lbw_review(TEST, "NOT_OUT", r, genuine_attempt_to_play=False).outcome == ReviewOutcome.OVERTURNED
    # Unknown attempt and it matters: unresolved.
    d = lbw_review(TEST, "NOT_OUT", r)
    assert d.outcome == ReviewOutcome.INCONCLUSIVE
    assert next(c for c in d.components if c.name == "impact").status == ComponentStatus.UNRESOLVED


def test_inconclusive_verdict_never_overturns():
    d = lbw_review(TEST, "NOT_OUT", result(verdict="INCONCLUSIVE"))
    assert d.outcome == ReviewOutcome.INCONCLUSIVE and d.final_decision == "NOT_OUT" and d.review_retained is None
    assert any("third umpire" in w for w in d.warnings)
    d = lbw_review(TEST, "OUT", result(leg=1.0, verdict="INSUFFICIENT_DATA"))
    assert d.outcome == ReviewOutcome.INCONCLUSIVE and d.final_decision == "OUT"


def test_uncertain_component_is_unresolved():
    d = lbw_review(TEST, "NOT_OUT", result(p_hit=0.7, pc=0.6, verdict="PROJECTED_TO_HIT"))
    assert d.outcome == ReviewOutcome.INCONCLUSIVE


def test_unknown_facts_block_overturn_to_out_but_failed_fact_gives_not_out():
    unknown = [{"id": "pitching", "status": "PASS"}]
    d = lbw_review(TEST, "NOT_OUT", result(checks=unknown))
    assert d.outcome == ReviewOutcome.INCONCLUSIVE and d.final_decision == "NOT_OUT"
    edged = [*FACTS_OK[:1], {"id": "not_bat_first", "status": "FAIL"}, *FACTS_OK[2:]]
    d = lbw_review(TEST, "OUT", result(checks=edged))
    assert d.outcome == ReviewOutcome.OVERTURNED and d.final_decision == "NOT_OUT"


def test_distance_rule():
    assert lbw_review(TEST, "NOT_OUT", result(dist=3.3)).outcome == ReviewOutcome.UMPIRES_CALL
    assert lbw_review(TEST, "NOT_OUT", result(dist=2.7, pti=0.2)).outcome == ReviewOutcome.UMPIRES_CALL
    assert lbw_review(TEST, "NOT_OUT", result(dist=2.7, pti=1.0)).outcome == ReviewOutcome.OVERTURNED
    assert lbw_review(TEST, "NOT_OUT", result(dist=3.0)).outcome == ReviewOutcome.INCONCLUSIVE  # straddles 3 m
    # Applies only to on-field NOT OUT.
    assert lbw_review(TEST, "OUT", result(dist=3.3)).outcome == ReviewOutcome.UPHELD
    assert lbw_review(CLUB, "NOT_OUT", result(dist=3.3)).outcome == ReviewOutcome.OVERTURNED


def test_full_toss_pitching_not_applicable():
    checks = [*FACTS_OK[:3], {"id": "pitching", "status": "NOT_APPLICABLE"}]
    d = lbw_review(TEST, "NOT_OUT", result(pitched=False, checks=checks))
    assert next(c for c in d.components if c.name == "pitching").status == ComponentStatus.NOT_APPLICABLE
    assert d.outcome == ReviewOutcome.OVERTURNED


def test_zone_probabilities_geometry():
    hw, r = 0.1143, 0.0364
    x = np.array([0.0, hw + r / 2, hw + 2 * r, -(hw + r / 2), -(hw + 2 * r), np.nan])
    z = zone_probabilities(x, x, np.full(6, 1.5), np.full(6, 3.0), hw, r, "right")
    imp = z["impact"]
    assert imp["centre_in_line"] == pytest.approx(1 / 6)
    assert imp["any_part_in_line"] == pytest.approx(3 / 6)
    assert imp["umpires_call_band_off"] == pytest.approx(1 / 6) and imp["umpires_call_band_leg"] == pytest.approx(1 / 6)
    assert imp["no_part_outside_off"] == pytest.approx(1 / 6) and imp["no_part_outside_leg"] == pytest.approx(1 / 6)
    assert z["pitching"]["centre_outside_leg"] == pytest.approx(2 / 6)
    assert z["pitch_to_impact_m"]["mean"] == pytest.approx(1.5)
    left = zone_probabilities(x, x, np.full(6, 1.5), None, hw, r, "left")
    assert left["impact"]["no_part_outside_off"] == pytest.approx(1 / 6)
    assert left["pitching"]["centre_outside_leg"] == pytest.approx(2 / 6)  # mirror image: same counts here
    assert zone_probabilities(x, x, np.full(6, 1.5), None, hw, r, None)["impact"]["no_part_outside_off"] is None


def test_simple_review_mappings():
    assert simple_review(TEST, "NOT_OUT", "OUT_OF_GROUND").outcome == ReviewOutcome.OVERTURNED
    d = simple_review(TEST, "OUT", "OUT_OF_GROUND")
    assert d.outcome == ReviewOutcome.UPHELD and d.consumes_review
    d = simple_review(TEST, "NONE", "INCONCLUSIVE", review_kind="umpire")
    assert d.final_decision == "REFER_TO_THIRD_UMPIRE" and not d.consumes_review and d.review_retained is None
    d = simple_review(TEST, "NONE", "IN_GROUND", review_type="stumping", review_kind="umpire")
    assert d.final_decision == "NOT_OUT"
    assert simple_review(TEST, "FOUR", "SIX", review_type="boundary", review_kind="umpire").final_decision == "SIX"
    with pytest.raises(ValueError):
        simple_review(TEST, "OUT", "OUT", review_type="timed_out")
    with pytest.raises(ValueError):
        simple_review(TEST, "NONE", "OUT")  # player review needs an on-field decision


def test_review_counts():
    reviews = [
        {"team": "A", "innings": 1, "kind": "player", "outcome": "UPHELD"},
        {"team": "A", "innings": 1, "kind": "player", "outcome": "UMPIRES_CALL"},
        {"team": "A", "innings": 1, "kind": "player", "outcome": "OVERTURNED"},
        {"team": "A", "innings": 1, "kind": "umpire", "outcome": "UPHELD"},
        {"team": "B", "innings": 2, "kind": "player", "outcome": "UPHELD"},
        {"team": "B", "innings": 2, "kind": "player", "outcome": "UPHELD"},
        {"team": "B", "innings": 2, "kind": "player", "outcome": "UPHELD"},
        {"team": "B", "innings": 2, "kind": "player", "outcome": "UPHELD"},
        {"team": "A", "innings": 3, "kind": "player", "outcome": None},
    ]
    c = review_counts(TEST, reviews)
    assert c["A"]["innings"] == 3 and c["A"]["remaining"] == 3 and c["A"]["pending"] == 1
    a1 = c["A"]["by_innings"][1]
    assert a1["unsuccessful_used"] == 1 and a1["retained_umpires_call"] == 1 and a1["successful"] == 1 and a1["remaining"] == 2
    assert c["B"]["remaining"] == 0
    club = review_counts(CLUB, reviews[:2])  # umpire's call does not retain under the club protocol
    assert club["A"]["unsuccessful_used"] == 2 and club["A"]["remaining"] == 0
