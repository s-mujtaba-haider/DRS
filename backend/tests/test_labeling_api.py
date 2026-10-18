"""Ball-labelling workflow: plan, human labels via annotations, export as a YOLO set, then build-compatible."""

import csv

import pytest
from fastapi.testclient import TestClient

from ml.evaluation.synthetic import make_delivery, write_synthetic_video


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    ann = tmp_path_factory.mktemp("annotations")
    mp.setenv("DRS_STORAGE_DIR", str(tmp_path_factory.mktemp("storage")))
    mp.setenv("DRS_ANNOTATIONS_DIR", str(ann))
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    from backend.app.main import app

    with TestClient(app) as c:
        yield c, ann
    mp.undo()
    get_settings.cache_clear()


def test_label_and_export(env, tmp_path):
    client, ann = env
    d = make_delivery(fps=50)
    p = tmp_path / "label me.mp4"
    write_synthetic_video(str(p), d, fps=50, lead_frames=0, tail_frames=0)
    with open(p, "rb") as fh:
        v = client.post("/api/videos", files={"file": ("label me.mp4", fh, "video/mp4")}).json()

    plan = client.get(f"/api/labeling/videos/{v['id']}/plan?step=4").json()
    assert plan["frames"][:3] == [0, 4, 8] and plan["labelled"] == {}
    assert client.get(f"/api/labeling/videos/{v['id']}/plan?step=0").status_code == 422

    x, y = d.uv[4]
    client.post(f"/api/videos/{v['id']}/annotations", json={"frame_index": 4, "action": "add", "x": x, "y": y, "radius_px": 6})
    client.post(f"/api/videos/{v['id']}/annotations", json={"frame_index": 8, "action": "add", "x": 10, "y": 10})
    client.post(f"/api/videos/{v['id']}/annotations", json={"frame_index": 8, "action": "mark_absent"})  # latest wins
    plan = client.get(f"/api/labeling/videos/{v['id']}/plan?step=4").json()
    assert plan["labelled"] == {"4": "ball", "8": "no_ball"}

    assert client.post("/api/labeling/export", json={"video_ids": [v["id"]], "name": "../x"}).status_code == 422
    r = client.post("/api/labeling/export", json={"video_ids": [v["id"]], "name": "ui_set"})
    assert r.status_code == 201, r.text
    assert (r.json()["positives"], r.json()["negatives"]) == (1, 1)
    out = ann / "ui_set"
    labels = sorted((out / "labels").glob("*.txt"))
    assert len(labels) == 2 and len(list((out / "images").glob("*.png"))) == 2
    pos = next(lab for lab in labels if lab.read_text().strip())
    cls, cx, cy, w, h = pos.read_text().split()
    assert cls == "0" and float(cx) == pytest.approx(x / 1280, abs=1e-4) and float(w) == pytest.approx(12 / 1280, abs=1e-4)
    prov = list(csv.DictReader((out / "provenance.csv").open()))
    assert {row["label_origin"] for row in prov} == {"human"} and len({row["source_sequence"] for row in prov}) == 1

    # Re-export replaces this video's rows instead of duplicating them.
    client.post("/api/labeling/export", json={"video_ids": [v["id"]], "name": "ui_set"})
    assert len(list(csv.DictReader((out / "provenance.csv").open()))) == 2
