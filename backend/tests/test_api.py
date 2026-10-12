import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.calibration.pitch_geometry import standard_landmarks
from ml.evaluation.synthetic import make_delivery, write_synthetic_video

CAL_POINTS = ["striker_off_stump_base_rh", "striker_leg_stump_base_rh", "striker_off_stump_top_rh",
              "striker_leg_stump_top_rh", "striker_middle_stump_top", "striker_popping_crease_x_plus_1m32",
              "striker_popping_crease_x_minus_1m32", "striker_middle_stump_base",
              "striker_bowling_crease_right", "striker_bowling_crease_left"]


@pytest.fixture(scope="module")
def delivery():
    return make_delivery(line_x=0.0, end_x=0.02, pitch_y=4.5, fps=50)


@pytest.fixture(scope="module")
def video_path(tmp_path_factory, delivery):
    p = tmp_path_factory.mktemp("v") / "synthetic_delivery.mp4"
    write_synthetic_video(str(p), delivery, fps=50, lead_frames=8, tail_frames=5)
    return p


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    storage = tmp_path_factory.mktemp("storage")
    monkeypatch_module.setenv("DRS_STORAGE_DIR", str(storage))
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    from backend.app.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _upload(client, path, name="clip.mp4"):
    with open(path, "rb") as fh:
        return client.post("/api/videos", files={"file": (name, fh, "video/mp4")})


def _wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/analysis/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)


def test_health_and_status(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    st = client.get("/api/system/status").json()
    assert "torch" in st and "effective_detector_mode" in st
    assert client.get("/api/health").headers["X-Request-ID"]


def test_datasets_registry(client):
    reg = client.get("/api/datasets").json()
    assert any(d["id"] == "hf-cricket-synth" for d in reg["datasets"])


def test_upload_validation(client, tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("hello")
    assert _upload(client, bad, "x.txt").status_code == 415
    empty = tmp_path / "e.mp4"
    empty.write_bytes(b"")
    assert _upload(client, empty, "e.mp4").status_code == 400
    junk = tmp_path / "j.mp4"
    junk.write_bytes(b"\x00junk" * 1000)
    assert _upload(client, junk, "j.mp4").status_code == 422


def test_path_traversal_name_is_sanitised(client, video_path):
    r = _upload(client, video_path, "../../evil.mp4")
    assert r.status_code == 201
    assert r.json()["original_filename"] == "evil.mp4"
    assert "/" not in r.json()["stored_name"] and ".." not in r.json()["stored_name"]


def test_job_validation(client):
    assert client.post("/api/analysis/jobs", json={"video_id": "nope"}).status_code == 404
    vid = client.get("/api/videos").json()[0]["id"]
    assert client.post("/api/analysis/jobs", json={"video_id": vid, "detector": "magic"}).status_code == 422
    assert client.post("/api/analysis/jobs", json={"video_id": vid, "start_frame": 10, "end_frame": 5}).status_code == 422


def test_end_to_end_tracking_calibration_lbw(client, video_path, delivery):
    v = _upload(client, video_path).json()
    assert v["meta"]["width"] == 1280 and v["meta"]["nominal_fps"] == pytest.approx(50)
    job = client.post("/api/analysis/jobs", json={"video_id": v["id"], "detector": "motion"}).json()
    assert job["status"] == "queued"
    j = _wait(client, job["id"])
    assert j["status"] == "succeeded", j.get("error")
    res = client.get(f"/api/analysis/jobs/{job['id']}/results").json()
    primary = next(t for t in res["tracking"]["tracks"] if t["is_primary"])
    obs = [p for p in primary["points"] if p["kind"] == "observed"]
    assert len(obs) >= 0.7 * len(delivery.times)
    # Observed positions match the synthetic ground truth (lead-in of 8 frames).
    errs = [np.hypot(p["x"] - delivery.uv[p["frame_index"] - 8][0], p["y"] - delivery.uv[p["frame_index"] - 8][1])
            for p in obs if 0 <= p["frame_index"] - 8 < len(delivery.uv)]
    assert np.median(errs) < 3.0

    for name in ("annotated.mp4", "track_points.csv", "summary.json"):
        r = client.get(f"/api/analysis/jobs/{job['id']}/artifacts/{name}")
        assert r.status_code == 200 and len(r.content) > 0
    assert client.get(f"/api/analysis/jobs/{job['id']}/artifacts/secret.txt").status_code == 404
    csv_head = client.get(f"/api/analysis/jobs/{job['id']}/artifacts/track_points.csv").text.splitlines()[0]
    assert csv_head.startswith("track_id,is_primary,frame_index,timestamp_ms,kind")

    # Calibration from landmark clicks (exact projections of the synthetic camera, known intrinsics).
    lm = standard_landmarks()
    cam = delivery.camera
    pts = [{"name": n, "image": cam.project(np.array([lm[n]]))[0].tolist()} for n in CAL_POINTS]
    cal = client.post("/api/calibration", json={"video_id": v["id"], "points": pts,
                                                "intrinsics": {"K": cam.K, "validated": True}})
    assert cal.status_code == 201, cal.text
    assert cal.json()["camera"]["reprojection_rms_px"] < 0.5

    # Manual correction is stored separately from model output.
    f0 = obs[3]["frame_index"]
    a = client.post(f"/api/videos/{v['id']}/annotations", json={"frame_index": f0, "action": "reject", "job_id": job["id"]})
    assert a.status_code == 201
    assert len(client.get(f"/api/videos/{v['id']}/annotations").json()) == 1

    lbw = client.post("/api/lbw/analyze", json={
        "job_id": job["id"], "calibration_id": cal.json()["id"],
        "facts": {"batter_handedness": "right"}}).json()
    r = lbw["result"]
    assert r["verdict"] in ("PROJECTED_TO_HIT", "INCONCLUSIVE", "PROJECTED_TO_MISS", "INSUFFICIENT_DATA")
    assert r["disclaimer"].startswith("Experimental")
    assert r["reconstruction_mode"] == "monocular_approximate"
    assert "1 manual correction(s) applied" in r["warnings"]
    assert r["overlay"]["wicket_outline_px"]

    pdf = client.get(f"/api/analysis/jobs/{job['id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    exp = client.get(f"/api/analysis/jobs/{job['id']}/export.json").json()
    assert exp["lbw_analyses"] and exp["tracking"]["tracks"]

    contact = client.post("/api/contact/analyze", json={"job_id": job["id"]}).json()
    assert contact["status"] == "INCONCLUSIVE"  # 50 fps is too slow to resolve contact

    stats = client.get("/api/stats").json()
    assert stats["lbw_analyses"] >= 1 and stats["jobs_succeeded"] >= 1


def test_video_range_request(client, video_path):
    v = _upload(client, video_path).json()
    r = client.get(f"/api/videos/{v['id']}/file", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


def test_frame_endpoint(client, video_path):
    v = _upload(client, video_path).json()
    r = client.get(f"/api/videos/{v['id']}/frame/5")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert float(r.headers["X-Frame-Timestamp-Ms"]) == pytest.approx(100.0, abs=1)
    assert client.get(f"/api/videos/{v['id']}/frame/99999").status_code == 404


def test_cancel_and_status_transitions(client, video_path):
    v = _upload(client, video_path).json()
    job = client.post("/api/analysis/jobs", json={"video_id": v["id"], "detector": "motion"}).json()
    client.post(f"/api/analysis/jobs/{job['id']}/cancel")
    j = _wait(client, job["id"])
    assert j["status"] in ("cancelled", "succeeded")  # may finish before the cancel lands
    if j["status"] == "cancelled":
        assert client.get(f"/api/analysis/jobs/{job['id']}/results").status_code == 409
    assert client.post(f"/api/analysis/jobs/{job['id']}/cancel").status_code == 409


def test_cricket_detector_missing_weights_fails_clearly(client, video_path):
    v = _upload(client, video_path).json()
    job = client.post("/api/analysis/jobs", json={"video_id": v["id"], "detector": "cricket"}).json()
    j = _wait(client, job["id"])
    assert j["status"] == "failed"
    assert "weights not found" in j["error"] or "PyTorch could not be loaded" in j["error"] or "ultralytics" in j["error"]


def test_calibration_errors(client):
    r = client.post("/api/calibration", json={"image_width": 100, "image_height": 100,
                                              "points": [{"name": "nope", "image": [1, 1]}] * 4})
    assert r.status_code == 422
    r = client.post("/api/calibration", json={"image_width": 100, "image_height": 100, "points": [{"image": [1, 1]}] * 4})
    assert r.status_code == 422


def test_triangulation_endpoint(client):
    from ml.calibration.camera import look_at_camera

    a = look_at_camera("A", np.array([0.5, 30.0, 6.0]), np.array([0, 2, 0.5]), 1500, 1920, 1080)
    b = look_at_camera("B", np.array([25.0, 3.0, 4.0]), np.array([0, 2, 0.5]), 1500, 1920, 1080)
    ids = [client.post("/api/calibration/import", json=c.model_dump(mode="json")).json()["id"] for c in (a, b)]
    obs = []
    for k in range(8):
        X = np.array([0.1, 12 - k, 1.5 - 0.1 * k])
        for c in (a, b):
            u, vv = c.project(X)[0]
            obs.append({"camera_id": c.camera_id, "frame_index": k, "timestamp_ms": k * 10.0, "u": u, "v": vv})
    r = client.post("/api/reconstruction/triangulate", json={"calibration_ids": ids, "observations": obs}).json()
    assert len(r["points"]) == 8
    assert np.allclose(r["points"][3]["xyz"], [0.1, 9, 1.2], atol=1e-3)
    assert "flight_fit" in r
