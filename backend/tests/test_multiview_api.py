"""Multi-view LBW and clip-group API (synthetic two-camera clip group; tracking results injected into job dirs)."""

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.calibration.camera import look_at_camera
from ml.common.schemas import BallTrack, TrackingResult, TrackPoint, TrackQuality
from ml.evaluation.synthetic import make_delivery
from ml.trajectory.ballistic import DEFAULT_DRAG_C

GROUP = "grp0123456789"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("DRS_STORAGE_DIR", str(tmp_path_factory.mktemp("storage")))
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    from backend.app.api import multiview
    from backend.app.main import app

    if not any(getattr(r, "path", None) == "/api/lbw/multiview" for r in app.routes):
        app.include_router(multiview.router)
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    mp.undo()


def _rig():
    a = look_at_camera("end_on", np.array([0.0, 42.0, 7.0]), np.array([0.0, 2.0, 0.4]), 4200.0, 1920, 1080)
    b = look_at_camera("side_on", np.array([32.0, 9.0, 4.0]), np.array([0.0, 8.0, 0.5]), 2600.0, 1920, 1080)
    for c in (a, b):
        c.validated = True
    return a, b


def _tracking(times_s, uv) -> TrackingResult:
    pts = [TrackPoint(frame_index=i, timestamp_ms=float(t * 1000), kind="observed", x=float(u), y=float(v))
           for i, (t, (u, v)) in enumerate(zip(times_s, uv))]
    q = TrackQuality(num_frames_spanned=len(pts), num_observed=len(pts), num_missing=0, observed_ratio=1.0, longest_gap=0,
                     mean_innovation_px=1.0, mean_confidence=0.9, reliable=True)
    return TrackingResult(tracks=[BallTrack(track_id=1, is_primary=True, points=pts, quality=q, score=1.0)],
                          primary_track_id=1, num_frames=len(pts), num_frames_with_candidates=len(pts), num_tracks_total=1)


@pytest.fixture(scope="module")
def clip_group(client):
    """Two captured clips (50 + 100 fps, offset clocks) with succeeded tracking jobs and imported calibrations."""
    from backend.app.core.config import get_settings
    from backend.app.database.models import Job, SessionLocal, Video

    d = make_delivery(pitch_y=5.0, line_x=0.0, end_x=0.0, speed_mps=34.0, fps=1000)
    model = replace(d.model, drag_c=DEFAULT_DRAG_C)
    ts = np.arange(0, 1.5, 1e-4)
    impact_t = float(ts[np.argmax(model.position(ts)[:, 1] <= 1.2)])
    rng = np.random.default_rng(0)
    out = {"impact_ms": impact_t * 1000.0 + 1000.0, "cams": []}
    cfg = get_settings()
    for cam, fps, phase, clip_offset_ms in zip(_rig(), (50.0, 100.0), (0.0, 0.003), (1000.0, 1000.0)):
        t_true = np.arange(phase, impact_t, 1 / fps)
        uv = cam.project(model.position(t_true)) + rng.normal(0, 0.5, (len(t_true), 2))
        # Clip-local time starts at 0 = shared time 1000 ms - clip_offset (offset maps local -> shared).
        t_local = t_true + (1000.0 - clip_offset_ms) / 1000.0
        with SessionLocal() as s:
            v = Video(original_filename=f"cap_{cam.camera_id}.mp4", stored_name="missing.mp4", size_bytes=1, sha256="0" * 64,
                      meta={"width": 1920, "height": 1080, "nominal_fps": fps,
                            "capture": {"camera_id": cam.camera_id, "clip_group": GROUP, "time_offset_ms": clip_offset_ms,
                                        "sync_uncertainty_ms": 0.0, "hardware_sync": False}})
            s.add(v)
            s.flush()
            j = Job(video_id=v.id, status="succeeded", config={"detector": "motion"})
            s.add(j)
            s.commit()
            vid, jid = v.id, j.id
        jd = cfg.jobs_dir / jid
        jd.mkdir(parents=True, exist_ok=True)
        (jd / "tracking.json").write_text(_tracking(t_local, uv).model_dump_json())
        cal = client.post(f"/api/calibration/import?video_id={vid}", json=cam.model_dump(mode="json"))
        assert cal.status_code == 201, cal.text
        out["cams"].append({"video_id": vid, "job_id": jid, "calibration_id": cal.json()["id"]})
    return out


def test_clip_group_listing_and_tracking(client, clip_group, monkeypatch):
    g = client.get(f"/api/clip-groups/{GROUP}").json()
    assert {v["camera_id"] for v in g["videos"]} == {"end_on", "side_on"}
    assert all(v["jobs"][0]["status"] == "succeeded" for v in g["videos"])
    assert client.get("/api/clip-groups/nope").status_code == 404

    submitted = []
    monkeypatch.setattr(client.app.state.jobs, "submit", submitted.append)
    r = client.post(f"/api/clip-groups/{GROUP}/track", json={"detector": "motion"}).json()
    assert r["queued"] == [] and len(r["skipped"]) == 2  # already tracked
    r = client.post(f"/api/clip-groups/{GROUP}/track", json={"detector": "motion", "force": True}).json()
    assert len(r["queued"]) == 2 and len(submitted) == 2
    assert {q["config"]["detector"] for q in r["queued"]} == {"motion"}


def test_multiview_lbw_hardware_sync_and_review_attachment(client, clip_group):
    cams = [{"job_id": c["job_id"], "calibration_id": c["calibration_id"]} for c in clip_group["cams"]]
    r = client.post("/api/lbw/multiview", json={"cameras": cams, "impact_time_ms": clip_group["impact_ms"], "hardware_sync": True,
                                                "facts": {"batter_handedness": "right"}})
    assert r.status_code == 201, r.text
    a = r.json()
    res = a["result"]
    assert res["reconstruction_mode"] == "multiview_3d"
    assert res["verdict"] == "PROJECTED_TO_HIT", res["reasons"]
    assert res["multiview"]["n_points"] >= 10 and res["multiview"]["hardware_sync"] is True
    assert a["request"]["resolved_cameras"][0]["offset_source"] == "capture"
    assert res["overlay"]["wicket_outline_px"]

    # The stored analysis is an ordinary LBW analysis: it attaches to an LBW review.
    pid = client.get("/api/protocols").json()[0]["id"]
    m = client.post("/api/matches", json={"name": "m", "team_a": "A", "team_b": "B", "protocol_id": pid}).json()
    rv = client.post(f"/api/matches/{m['id']}/reviews", json={"requested_by": "B", "decision_type": "lbw", "on_field": "NOT_OUT"}).json()
    assert client.post(f"/api/reviews/{rv['id']}/evidence", json={"type": "lbw", "id": a["id"]}).status_code == 200


def test_multiview_software_sync_is_inconclusive(client, clip_group):
    cams = [{"job_id": c["job_id"], "calibration_id": c["calibration_id"], "sync_uncertainty_ms": 15.0} for c in clip_group["cams"]]
    res = client.post("/api/lbw/multiview", json={"cameras": cams, "impact_time_ms": clip_group["impact_ms"],
                                                  "facts": {"batter_handedness": "right"}}).json()["result"]
    assert res["verdict"] == "INCONCLUSIVE"
    assert res["multiview"]["sync_definite_allowed"] is False


def test_multiview_validation(client, clip_group):
    c0, c1 = clip_group["cams"]
    one = {"job_id": c0["job_id"], "calibration_id": c0["calibration_id"]}
    assert client.post("/api/lbw/multiview", json={"cameras": [one]}).status_code == 422
    assert client.post("/api/lbw/multiview", json={"cameras": [one, one]}).status_code == 422
    missing = {"job_id": "nope", "calibration_id": c1["calibration_id"]}
    assert client.post("/api/lbw/multiview", json={"cameras": [one, missing]}).status_code == 404
