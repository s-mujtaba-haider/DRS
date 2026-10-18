"""TensorRT engine selection: validation sidecar checks, input-profile fallback and agreement metrics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ml.detection.yolo_detector import YoloBallDetector, file_sha256, validated_engine

ROOT = Path(__file__).resolve().parents[1]
HAS_TRT = importlib.util.find_spec("tensorrt") is not None


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _export_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import export_tensorrt
    finally:
        sys.path.pop(0)
    return export_tensorrt


def test_compare_agreement_metrics():
    m = _export_module()
    ref = [[(10, 10, 20, 20, 0.9), (50, 50, 60, 60, 0.3)], [(0, 0, 8, 8, 0.8)], []]
    eng = [[(10.5, 10, 20.5, 20, 0.89), (50, 50, 60, 60, 0.31), (90, 90, 95, 95, 0.5)], [], [(1, 1, 3, 3, 0.1)]]
    c = m.compare(ref, eng)
    assert c["reference_detections"] == 3 and c["engine_detections"] == 3 and c["matched"] == 2
    assert c["agreement"] == pytest.approx(2 / 3)
    assert c["engine_extra_rate"] == pytest.approx(1 / 3)  # (90,90) box; the 0.1-score box is below threshold
    assert c["centre_err_px_median"] == pytest.approx(0.25)


@pytest.mark.skipif(not (HAS_TRT and _cuda()), reason="TensorRT and CUDA required")
def test_validated_engine_sidecar_checks(tmp_path):
    import tensorrt
    import torch

    pt, eng, side = tmp_path / "m.pt", tmp_path / "m.engine", tmp_path / "m.engine.json"
    pt.write_bytes(b"weights")
    assert validated_engine(pt)[0] is None  # no engine at all
    eng.write_bytes(b"engine")
    assert "no validation sidecar" in validated_engine(pt)[1]
    good = {"source_sha256": file_sha256(pt), "engine_sha256": file_sha256(eng), "tensorrt": tensorrt.__version__,
            "gpu": torch.cuda.get_device_name(0), "validation": {"passed": True}}
    side.write_text(json.dumps(good))
    assert validated_engine(pt)[0] == eng
    for key, bad, msg in [("validation", {"passed": False}, "did not pass"), ("tensorrt", "0.0", "built with TensorRT"),
                          ("source_sha256", "x", "different .pt"), ("engine_sha256", "x", "engine file differs"),
                          ("gpu", "Other GPU", "built for GPU")]:
        side.write_text(json.dumps({**good, key: bad}))
        path, note = validated_engine(pt)
        assert path is None and msg in note, (key, note)


def test_letterbox_fit_rule():
    d = YoloBallDetector.__new__(YoloBallDetector)
    d._engine_max_hw = (736, 1280)
    im = lambda h, w: np.zeros((h, w, 3), np.uint8)  # noqa: E731
    assert d._fits_engine(im(1080, 1920), 1280)  # 16:9 -> 720 -> 736
    assert d._fits_engine(im(720, 1280), 1280)
    assert not d._fits_engine(im(960, 1280), 1280)  # 4:3 -> 960
    assert not d._fits_engine(im(1920, 1080), 1280)  # portrait


ENGINE = ROOT / "models/checkpoints/cricket_ball_best.engine"


@pytest.mark.skipif(not (HAS_TRT and _cuda() and ENGINE.is_file()), reason="validated engine not built on this machine")
def test_engine_matches_pytorch_and_falls_back_outside_profile():
    from ml.detection.base import FrameInfo
    from ml.video.ingest import iter_frames

    w = ROOT / "models/checkpoints/cricket_ball_best.pt"
    trt = YoloBallDetector(w, prefer_engine=True, batch_size=4)
    if not trt.uses_engine:
        pytest.skip(trt.engine_note or "engine not validated")
    assert trt.model_id.endswith(":trt-fp16")
    clip = ROOT / "data/raw/hf-jeffrey632-delivery-analysis/example-03.mp4"
    if not clip.is_file():
        pytest.skip("benchmark clip not downloaded")
    frames = [r.image for r in iter_frames(clip, max_frames=8)]
    infos = [FrameInfo(i, i * 40.0, "container_pts") for i in range(8)]
    pt = YoloBallDetector(w, batch_size=4)
    a, b = pt.detect_batch(frames, infos), trt.detect_batch(frames, infos)
    for da, db in zip(a, b):
        sa = [d for d in da if d.confidence >= 0.25]
        sb = [d for d in db if d.confidence >= 0.25]
        assert len(sa) == len(sb)
        for x, y in zip(sa, sb):
            assert np.hypot(x.center[0] - y.center[0], x.center[1] - y.center[1]) < 1.0
    # A 4:3 frame does not fit the 16:9 profile: it must be served by the .pt model, not resized.
    sq = np.zeros((960, 1280, 3), np.uint8)
    trt.detect_batch([sq], [FrameInfo(0, 0.0, "container_pts")])
    assert trt._pt_model is not None and trt.uses_engine
