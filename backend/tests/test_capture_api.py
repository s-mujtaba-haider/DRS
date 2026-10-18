"""Live capture through the API with synthetic cameras: start, preview, clip into a review, sync, stop."""

import time

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


def test_capture_validation(client):
    assert client.get("/api/capture/status").json()["cameras"] == []
    assert client.post("/api/capture/clip", json={"seconds": 2}).status_code == 409
    bad = {"cameras": [{"camera_id": "a", "type": "opencv", "uri": "C:/Windows/win.ini"}]}
    assert client.post("/api/capture/start", json=bad).status_code == 422  # no local file paths
    dup = {"cameras": [{"camera_id": "a", "type": "synthetic"}, {"camera_id": "a", "type": "synthetic"}]}
    assert client.post("/api/capture/start", json=dup).status_code == 422
    assert client.post("/api/capture/start", json={"cameras": [{"camera_id": "../x", "type": "synthetic"}]}).status_code == 422


def test_capture_clip_into_review_and_sync(client):
    cams = [{"camera_id": cid, "type": "synthetic", "width": 320, "height": 240, "fps": 50} for cid in ("end_on", "side_on")]
    r = client.post("/api/capture/start", json={"cameras": cams})
    assert r.status_code == 200, r.text
    assert r.json()["hardware_sync"] is False
    assert client.post("/api/capture/start", json={"cameras": cams[:1]}).status_code == 409
    time.sleep(2.5)
    p = client.get("/api/capture/preview/end_on")
    assert p.status_code == 200 and p.content[:2] == b"\xff\xd8"
    assert client.get("/api/capture/preview/nope").status_code == 404

    pid = client.get("/api/protocols").json()[0]["id"]
    m = client.post("/api/matches", json={"name": "m", "team_a": "A", "team_b": "B", "protocol_id": pid}).json()
    rv = client.post(f"/api/matches/{m['id']}/reviews", json={"requested_by": "umpire", "decision_type": "run_out", "on_field": "NONE"}).json()

    clip = client.post("/api/capture/clip", json={"seconds": 1.5, "review_id": rv["id"], "label": "rv1"})
    assert clip.status_code == 201, clip.text
    c = clip.json()
    assert len(c["videos"]) == 2 and c["sync"]["hardware_sync"] is False
    for v in c["videos"]:
        cap = v["meta"]["capture"]
        assert cap["clip_group"] == c["clip_group"] and cap["sync_uncertainty_ms"] > 0
        assert 50 <= v["meta"]["frame_count_decoded"] <= 90  # ~1.5 s at 50 fps
        assert client.get(f"/api/videos/{v['id']}/frame/3").status_code == 200
    review = client.get(f"/api/reviews/{rv['id']}").json()
    assert set(review["video_ids"]) == {v["id"] for v in c["videos"]}
    assert "capture_clip_added" in [e["action"] for e in review["events"]]

    a, b = (v["id"] for v in c["videos"])
    ev = client.post("/api/sync/events", json={"reference_video_id": a, "other_video_id": b,
                                                "events_reference_ms": [100.0, 600.0], "events_other_ms": [90.0, 590.0]})
    assert ev.status_code == 200 and ev.json()["offset_ms"] == pytest.approx(10.0, abs=20.0)
    au = client.post("/api/sync/audio", json={"reference_video_id": a, "other_video_id": b})
    assert au.status_code == 200 and au.json()["status"] == "NO_AUDIO"

    st = client.post("/api/capture/stop").json()
    assert all(not cam["running"] for cam in st["cameras"])
