import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("DRS_STORAGE_DIR", str(tmp_path_factory.mktemp("storage")))
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    from backend.app.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    mp.undo()


@pytest.fixture(scope="module")
def protocol(client):
    ps = client.get("/api/protocols").json()
    assert ps, "no DRS protocols configured"
    p = next((p for p in ps if "t20" in p["id"].lower()), ps[0])
    return client.get(f"/api/protocols/{p['id']}").json()


@pytest.fixture()
def match(client, protocol):
    r = client.post("/api/matches", json={"name": "Test fixture", "team_a": "Lions", "team_b": "Tigers",
                                          "protocol_id": protocol["id"], "batting_team": "Lions",
                                          "officials": {"third_umpire": "TV Umpire"}})
    assert r.status_code == 201, r.text
    return r.json()


def _review(client, m, **kw):
    body = {"requested_by": "Tigers", "decision_type": "lbw", "on_field": "NOT_OUT", "over_ball": "3.2", **kw}
    return client.post(f"/api/matches/{m['id']}/reviews", json=body)


def test_match_validation(client, protocol):
    assert client.post("/api/matches", json={"name": "x", "team_a": "A", "team_b": "a", "protocol_id": protocol["id"]}).status_code == 422
    assert client.post("/api/matches", json={"name": "x", "team_a": "A", "team_b": "B", "protocol_id": "nope"}).status_code == 422


def test_review_lifecycle_and_counts(client, match, protocol):
    allowed = protocol["reviews"]["unsuccessful_reviews_per_innings"]
    start = match["reviews"]["teams"]["Tigers"]["remaining"]
    assert start == allowed

    r = _review(client, match)
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    assert r.json()["kind"] == "player" and r.json()["innings"] == 1
    assert _review(client, match).status_code == 409  # one open review per team

    rec = client.post(f"/api/reviews/{rid}/recommend").json()["recommendation"]
    assert rec["outcome"] == "INCONCLUSIVE" and rec["advisory"] is True  # no evidence attached

    bad = client.post(f"/api/reviews/{rid}/decide", json={"final_decision": "NOT_OUT", "outcome": "OVERTURNED", "decided_by": "TV"})
    assert bad.status_code == 422
    d = client.post(f"/api/reviews/{rid}/decide", json={"final_decision": "NOT_OUT", "outcome": "UPHELD", "decided_by": "TV"})
    assert d.status_code == 200 and d.json()["status"] == "decided"
    assert client.post(f"/api/reviews/{rid}/decide", json={"final_decision": "OUT", "outcome": "OVERTURNED", "decided_by": "TV"}).status_code == 409

    m = client.get(f"/api/matches/{match['id']}").json()
    assert m["reviews"]["teams"]["Tigers"]["remaining"] == allowed - 1
    assert m["reviews"]["teams"]["Lions"]["remaining"] == allowed

    # Umpire's call keeps the review; an overturn keeps it too.
    r2 = _review(client, match).json()["id"]
    client.post(f"/api/reviews/{r2}/decide", json={"final_decision": "NOT_OUT", "outcome": "UMPIRES_CALL", "decided_by": "TV"})
    r3 = _review(client, match).json()["id"]
    client.post(f"/api/reviews/{r3}/decide", json={"final_decision": "OUT", "outcome": "OVERTURNED", "decided_by": "TV"})
    c = client.get(f"/api/matches/{match['id']}").json()["reviews"]["teams"]["Tigers"]
    assert c["remaining"] == allowed - 1
    if protocol["reviews"]["umpires_call_retains_review"]:
        assert c["retained_umpires_call"] == 1

    # Exhaust the remaining reviews, then a new player review is refused.
    for _ in range(allowed - 1):
        x = _review(client, match).json()["id"]
        client.post(f"/api/reviews/{x}/decide", json={"final_decision": "NOT_OUT", "outcome": "UPHELD", "decided_by": "TV"})
    assert _review(client, match).status_code == 409

    # Umpire referrals never consume reviews and are always allowed.
    u = _review(client, match, requested_by="umpire", decision_type="run_out", on_field="NONE")
    assert u.status_code == 201 and u.json()["kind"] == "umpire"

    # New innings resets the per-innings allowance.
    m = client.patch(f"/api/matches/{match['id']}", json={"current_innings": 2, "batting_team": "Tigers"}).json()
    assert m["reviews"]["teams"]["Tigers"]["remaining"] == allowed


def test_review_rules_and_package(client, match):
    assert _review(client, match, requested_by="Nobody").status_code == 422
    assert _review(client, match, on_field="NONE").status_code == 422
    assert _review(client, match, decision_type="boundary", requested_by="umpire", on_field="NONE").status_code == 201
    assert _review(client, match, decision_type="boundary").status_code == 422  # not player-reviewable
    assert _review(client, match, over_ball="3.x").status_code == 422
    rid = _review(client, match, decision_type="stumping", on_field="OUT").json()["id"]
    assert client.post(f"/api/reviews/{rid}/decide", json={"final_decision": "OUT", "outcome": "UMPIRES_CALL", "decided_by": "TV"}).status_code == 422
    assert client.post(f"/api/reviews/{rid}/evidence", json={"type": "decision", "id": "missing"}).status_code == 404
    client.post(f"/api/reviews/{rid}/withdraw", json={"reason": "duplicate"})
    pkg = client.get(f"/api/reviews/{rid}/package").json()
    assert pkg["review"]["status"] == "withdrawn"
    assert [e["action"] for e in pkg["events"]] == ["created", "withdrawn"]
    assert pkg["protocol"]["id"] == match["protocol_id"] and "advisory" in pkg["disclaimer"]


def test_completed_match_refuses_reviews(client, match):
    client.patch(f"/api/matches/{match['id']}", json={"status": "completed"})
    assert _review(client, match).status_code == 409
