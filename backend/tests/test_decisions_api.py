"""API tests for run-out, wicket-break and boundary evidence, wired into a match review (synthetic footage)."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

BREAK_FRAME = 12


def stumps_frame(bail_off: bool, rng):
    img = np.full((240, 320, 3), (40, 110, 40), np.uint8)
    for x in (140, 158, 176):
        cv2.rectangle(img, (x, 80), (x + 6, 200), (235, 235, 235), -1)
    cv2.rectangle(img, (139, 74), (163, 78), (60, 200, 230), -1)
    if bail_off:
        cv2.rectangle(img, (200, 40), (224, 44), (60, 200, 230), -1)
    else:
        cv2.rectangle(img, (160, 74), (183, 78), (60, 200, 230), -1)
    cv2.line(img, (0, 210), (319, 210), (250, 250, 250), 2)  # crease marking
    return np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)


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
def video(client, tmp_path_factory):
    from ml.video.ingest import write_video

    rng = np.random.default_rng(1)
    p = write_video((stumps_frame(i >= BREAK_FRAME, rng) for i in range(30)), tmp_path_factory.mktemp("v") / "runout.mp4", 50)
    with open(p, "rb") as fh:
        r = client.post("/api/videos", files={"file": ("runout.mp4", fh, "video/mp4")})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture(scope="module")
def match(client):
    pid = client.get("/api/protocols").json()[0]["id"]
    return client.post("/api/matches", json={"name": "m", "team_a": "A", "team_b": "B", "protocol_id": pid}).json()


CREASE = {"source": "manual", "p1": [0, 209], "p2": [319, 209], "behind_point": [160, 230], "end": "striker"}


def test_wicket_break_detection(client, video):
    r = client.post("/api/wicket-break/detect", json={"video_id": video["id"], "roi": [130, 66, 230, 205]})
    assert r.status_code == 201, r.text
    d = r.json()["result"]
    assert d["status"] == "BREAK_CANDIDATE" and d["first_broken_frame"] == BREAK_FRAME
    assert client.post("/api/wicket-break/detect", json={"video_id": video["id"], "roi": [0, 0, 4, 4]}).status_code == 422
    assert client.post("/api/wicket-break/detect", json={"video_id": video["id"], "roi": [0, 0, 400, 200]}).status_code == 422


def _wb(client, video):
    d = client.post("/api/wicket-break/detect", json={"video_id": video["id"], "roi": [130, 66, 230, 205]}).json()["result"]
    return {k: d[k] for k in ("last_intact_frame", "last_intact_ms", "first_broken_frame", "first_broken_ms")} | {
        "source": "auto", "confirmed": True}


def _obs(frames, y, grounded):
    return [{"frame_index": f, "timestamp_ms": f * 20.0, "point": [100, y], "grounded": grounded, "point_sigma_px": 1.0} for f in frames]


def test_runout_in_and_out_of_ground_and_review(client, video, match):
    wb = _wb(client, video)
    facts = {"batter": "striker", "no_ball": False, "wicket_broken_fairly": True, "ball_touched_fielder_before_break": True,
             "left_ground_to_avoid_injury": False}
    safe = client.post("/api/runout/analyze", json={"video_id": video["id"], "dismissal": "run_out", "crease": CREASE,
                                                    "observations": _obs(range(6, 16), 225, True), "wicket_break": wb, "facts": facts})
    assert safe.status_code == 201, safe.text
    assert safe.json()["result"]["status"] == "IN_GROUND"

    out = client.post("/api/runout/analyze", json={"video_id": video["id"], "dismissal": "run_out", "crease": CREASE,
                                                   "observations": _obs(range(6, 16), 190, True), "wicket_break": wb, "facts": facts}).json()
    assert out["result"]["status"] == "OUT_OF_GROUND"
    assert out["result"]["crease_line_px"] == [[0, 209], [319, 209]]

    # Umpire referral, no on-field call: evidence drives the advisory decision.
    rv = client.post(f"/api/matches/{match['id']}/reviews", json={"requested_by": "umpire", "decision_type": "run_out",
                                                                   "on_field": "NONE", "video_ids": [video["id"]]}).json()
    assert client.post(f"/api/reviews/{rv['id']}/evidence", json={"type": "decision", "id": out["id"]}).status_code == 200
    rec = client.post(f"/api/reviews/{rv['id']}/recommend").json()["recommendation"]
    assert rec["final_decision"] == "OUT" and rec["evidence_id"] == out["id"] and rec["advisory"]
    pdf = client.get(f"/api/reviews/{rv['id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF" and b"/Subtype /Image" in pdf.content  # evidence frames

    # Validation: crease needs both points; points must be in frame.
    bad = client.post("/api/runout/analyze", json={"video_id": video["id"], "dismissal": "run_out",
                                                   "crease": {"source": "manual", "p1": [0, 0]}, "observations": _obs([1], 10, True)})
    assert bad.status_code == 422
    off = client.post("/api/runout/analyze", json={"video_id": video["id"], "dismissal": "run_out", "crease": CREASE,
                                                   "observations": _obs([1], 9999, True)})
    assert off.status_code == 422


def test_grounding_inside_break_window_is_inconclusive(client, video, match):
    wb = _wb(client, video)
    obs = _obs(range(6, BREAK_FRAME), 190, True) + _obs(range(BREAK_FRAME, 16), 225, True)
    r = client.post("/api/runout/analyze", json={"video_id": video["id"], "dismissal": "run_out", "crease": CREASE,
                                                 "observations": obs, "wicket_break": wb}).json()
    assert r["result"]["status"] == "INCONCLUSIVE"
    rv = client.post(f"/api/matches/{match['id']}/reviews", json={"requested_by": "umpire", "decision_type": "run_out",
                                                                   "on_field": "NONE"}).json()
    client.post(f"/api/reviews/{rv['id']}/evidence", json={"type": "decision", "id": r["id"]})
    rec = client.post(f"/api/reviews/{rv['id']}/recommend").json()["recommendation"]
    assert rec["outcome"] == "INCONCLUSIVE" and rec["final_decision"] == "REFER_TO_THIRD_UMPIRE"


def test_evidence_kind_must_match_review(client, video, match):
    a = client.post("/api/runout/analyze", json={"video_id": video["id"], "dismissal": "stumping", "crease": CREASE,
                                                 "observations": _obs([5], 225, True)}).json()
    rv = client.post(f"/api/matches/{match['id']}/reviews", json={"requested_by": "umpire", "decision_type": "boundary",
                                                                   "on_field": "NONE"}).json()
    assert client.post(f"/api/reviews/{rv['id']}/evidence", json={"type": "decision", "id": a["id"]}).status_code == 422


def test_boundary_six_and_review(client, video, match):
    rope = {"frame": "image", "points": [[0, 50], [319, 50]], "field_side_point": [160, 200]}
    r = client.post("/api/boundary/analyze", json={
        "video_id": video["id"], "rope": rope, "facts": {"struck_by_bat": True},
        "events": [{"kind": "ball_ground", "timestamp_ms": 500, "position": {"point": [160, 20], "frame": "image"}}]})
    assert r.status_code == 201, r.text
    assert r.json()["result"]["status"] == "BOUNDARY_SIX"
    rv = client.post(f"/api/matches/{match['id']}/reviews", json={"requested_by": "umpire", "decision_type": "boundary",
                                                                   "on_field": "NONE"}).json()
    client.post(f"/api/reviews/{rv['id']}/evidence", json={"type": "decision", "id": r.json()["id"]})
    rec = client.post(f"/api/reviews/{rv['id']}/recommend").json()["recommendation"]
    assert rec["final_decision"] == "SIX" and rec["boundary_status"] == "BOUNDARY_SIX"
    d = client.post(f"/api/reviews/{rv['id']}/decide", json={"boundary_result": "SIX", "outcome": "UPHELD", "decided_by": "TV"})
    assert d.status_code == 200 and d.json()["final_decision"] == "SIX"
    pdf = client.get(f"/api/reviews/{rv['id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    bad = client.post("/api/boundary/analyze", json={"rope": rope, "events": [{"kind": "teleport", "timestamp_ms": 1}]})
    assert bad.status_code == 422


def test_audio_contact_endpoint(client, match, tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from test_audio_contact import write_av_clip

    p = write_av_clip(tmp_path / "edge.mp4", click_ms=600.0)
    with open(p, "rb") as fh:
        v = client.post("/api/videos", files={"file": ("edge.mp4", fh, "video/mp4")}).json()
    r = client.post("/api/contact/audio", json={"video_id": v["id"], "center_ms": 600, "half_window_ms": 300})
    assert r.status_code == 201, r.text
    res = r.json()["result"]
    assert res["status"] == "TRANSIENTS_FOUND" and res["transients"][0]["nearest_frame_index"] in (29, 30)
    rv = client.post(f"/api/matches/{match['id']}/reviews", json={"requested_by": "A", "decision_type": "caught", "on_field": "NOT_OUT"}).json()
    assert client.post(f"/api/reviews/{rv['id']}/evidence", json={"type": "decision", "id": r.json()["id"]}).status_code == 200
    silent = client.post("/api/contact/audio", json={"video_id": client.get("/api/videos").json()[-1]["id"]}).json()
    assert silent["result"]["status"] == "NO_AUDIO"


def test_no_ball_check_nullifies_lbw(client, video, match):
    crease = {"source": "manual", "p1": [0, 150], "p2": [319, 150], "behind_point": [160, 200], "end": "bowler"}
    nb = client.post("/api/noball/analyze", json={"video_id": video["id"], "popping_crease": crease,
                                                 "front_marks": [{"frame_index": 5, "timestamp_ms": 100, "point": [120, 140], "sigma_px": 1}]})
    assert nb.status_code == 201, nb.text
    assert nb.json()["result"]["status"] == "NO_BALL"
    fair = client.post("/api/noball/analyze", json={"video_id": video["id"], "popping_crease": crease,
                                                   "front_marks": [{"frame_index": 5, "timestamp_ms": 100, "point": [120, 165], "sigma_px": 1}]})
    assert fair.json()["result"]["status"] == "FAIR"

    rv = client.post(f"/api/matches/{match['id']}/reviews", json={"requested_by": "B", "decision_type": "lbw", "on_field": "OUT"}).json()
    assert client.post(f"/api/reviews/{rv['id']}/evidence", json={"type": "decision", "id": nb.json()["id"]}).status_code == 200
    rec = client.post(f"/api/reviews/{rv['id']}/recommend").json()["recommendation"]
    assert rec["final_decision"] == "NOT_OUT" and rec["outcome"] == "OVERTURNED"
    assert any("No ball" in x for x in rec["reasons"])
    client.post(f"/api/reviews/{rv['id']}/withdraw", json={"reason": "test"})

    assert client.post(f"/api/matches/{match['id']}/reviews", json={"requested_by": "A", "decision_type": "no_ball", "on_field": "OUT"}).status_code == 422
    chk = client.post(f"/api/matches/{match['id']}/reviews", json={"requested_by": "umpire", "decision_type": "no_ball", "on_field": "OUT"}).json()
    client.post(f"/api/reviews/{chk['id']}/evidence", json={"type": "decision", "id": fair.json()["id"]})
    rec2 = client.post(f"/api/reviews/{chk['id']}/recommend").json()["recommendation"]
    assert rec2["final_decision"] == "OUT"  # fair delivery: the dismissal stands
    pdf = client.get(f"/api/reviews/{chk['id']}/report.pdf")
    assert pdf.status_code == 200 and b"/Subtype /Image" in pdf.content  # annotated landing frame embedded
