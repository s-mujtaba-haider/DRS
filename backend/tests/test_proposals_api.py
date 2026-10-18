"""POST /api/runout/proposals with a fake pose model (no weights needed)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("DRS_STORAGE_DIR", str(tmp_path_factory.mktemp("storage")))
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    from backend.app.api import proposals
    from backend.app.main import app

    if not any(getattr(r, "path", "") == "/api/runout/proposals" for r in app.routes):
        app.include_router(proposals.router)
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    mp.undo()


class FakePose:
    model_id = "fake-pose"

    def predict(self, images):
        from ml.runout.proposals import L_ANKLE, R_ANKLE, PersonPose

        out = []
        for _ in images:
            kp = np.zeros((17, 3))
            kp[:, 2] = 0.9
            kp[:, :2] = (160, 150)
            kp[L_ANKLE] = (150, 200, 0.9)
            kp[R_ANKLE] = (170, 200, 0.9)
            out.append([PersonPose((130, 80, 190, 205), 0.9, kp)])
        return out


@pytest.fixture(scope="module")
def video(client, tmp_path_factory):
    from ml.video.ingest import write_video

    p = write_video((np.full((240, 320, 3), 80, np.uint8) for _ in range(12)), tmp_path_factory.mktemp("v") / "p.mp4", 50)
    with open(p, "rb") as fh:
        return client.post("/api/videos", files={"file": ("p.mp4", fh, "video/mp4")}).json()


def test_proposals_endpoint(client, video, monkeypatch):
    from backend.app.api import proposals

    monkeypatch.setattr(proposals, "get_pose_model", lambda: FakePose())
    r = client.post("/api/runout/proposals", json={"video_id": video["id"], "start_frame": 2, "end_frame": 8,
                                                   "crease": {"p1": [0, 200], "p2": [320, 200]}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model_id"] == "fake-pose" and "AGPL" in d["licence"] and len(d["frames"]) == 6
    assert d["frames_with_batter"] == 6
    p = d["frames"][2]["proposals"][0]
    assert p["source"] == "model" and p["kind"] == "ankle_left" and p["grounded_hint"] is True


def test_proposals_validation(client, video, monkeypatch):
    from backend.app.api import proposals
    from ml.runout.proposals import PoseUnavailableError

    monkeypatch.setattr(proposals, "get_pose_model", lambda: FakePose())
    base = {"video_id": video["id"], "start_frame": 0, "end_frame": 5, "roi": [100, 50, 220, 230]}
    assert client.post("/api/runout/proposals", json={k: v for k, v in base.items() if k != "roi"}).status_code == 422
    assert client.post("/api/runout/proposals", json={**base, "end_frame": 0}).status_code == 422
    assert client.post("/api/runout/proposals", json={**base, "end_frame": 1000}).status_code == 422
    assert client.post("/api/runout/proposals", json={**base, "roi": [10, 10, 5, 50]}).status_code == 422
    assert client.post("/api/runout/proposals", json={**base, "roi": [0, 0, 9999, 50]}).status_code == 422
    assert client.post("/api/runout/proposals", json={**base, "video_id": "nope"}).status_code == 404

    def boom():
        raise PoseUnavailableError("Pose weights not found")

    monkeypatch.setattr(proposals, "get_pose_model", boom)
    r = client.post("/api/runout/proposals", json=base)
    assert r.status_code == 503 and "weights" in r.json()["detail"]
