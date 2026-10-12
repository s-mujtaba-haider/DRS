import numpy as np
import pytest
from pydantic import ValidationError

from ml.common.schemas import Detection
from ml.detection.base import FrameInfo, ModelUnavailableError, make_detection, nms, tile_grid
from ml.detection.motion_detector import MotionCandidateDetector


def test_detection_valid():
    d = Detection(frame_index=120, timestamp_ms=1000.0, bbox=(420, 280, 436, 296), center=(428, 288),
                  confidence=0.9, model_id="m")
    assert d.center == (428, 288)


@pytest.mark.parametrize("bbox,center", [((10, 10, 5, 20), (7, 15)), ((0, 0, 10, 10), (20, 5))])
def test_detection_invalid_geometry(bbox, center):
    with pytest.raises(ValidationError):
        Detection(frame_index=0, timestamp_ms=0, bbox=bbox, center=center, confidence=0.5, model_id="m")


def test_detection_confidence_bounds():
    with pytest.raises(ValidationError):
        Detection(frame_index=0, timestamp_ms=0, bbox=(0, 0, 2, 2), center=(1, 1), confidence=1.5, model_id="m")


def test_make_detection_clamps_and_centres():
    d = make_detection(FrameInfo(3, 60.0, "container_pts"), (10, 10, 10.2, 18), 1.2, "m", "model")
    assert d.confidence == 1.0
    assert d.bbox[2] - d.bbox[0] >= 1.0
    assert d.center == ((d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2)


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep == [0, 2]


def test_tile_grid_covers_frame():
    tiles = tile_grid(1920, 1080, 640, 0.2)
    assert min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 1920
    assert max(t[3] for t in tiles) == 1080
    assert tile_grid(320, 240, 640, 0.2) == [(0, 0, 320, 240)]


def test_motion_detector_finds_moving_blob():
    det = MotionCandidateDetector(compensate_camera_motion=False)
    found = []
    for i in range(6):
        img = np.full((240, 320, 3), 90, np.uint8)
        x = 40 + 30 * i
        img[100:106, x : x + 6] = 250
        found.append(det.detect_frame(img, FrameInfo(i, i * 20.0, "container_pts")))
    assert not found[0] and not found[1]  # needs two frames of history
    for i in range(2, 6):
        assert found[i], f"no candidate at frame {i}"
        cx, cy = found[i][0].center
        assert abs(cx - (43 + 30 * i)) < 4 and abs(cy - 103) < 4


def test_missing_weights_raise_clear_error(tmp_path):
    pytest.importorskip("ultralytics")
    from ml.detection.yolo_detector import YoloBallDetector

    with pytest.raises(ModelUnavailableError, match="not found"):
        YoloBallDetector(tmp_path / "missing.pt")


def test_split_by_sequence_keeps_sequences_whole_and_fills_splits():
    from ml.datasets.yolo_tools import split_by_sequence

    seqs = {f"clip{i}": [f"clip{i}_f{j}" for j in range(10 + i)] for i in range(3)}
    out = split_by_sequence(seqs)
    for items in seqs.values():
        assert len({out[x] for x in items}) == 1  # no leakage within a sequence
    assert set(out.values()) == {"train", "val", "test"}


def test_detection_metrics_centre_matching():
    from ml.evaluation.detection_metrics import match_frame, summarize

    gts = [(100, 100, 108, 108)]
    m1 = match_frame([(101, 101, 109, 109, 0.9), (300, 300, 308, 308, 0.8)], gts)
    m2 = match_frame([], [(10, 10, 16, 16)])
    s = summarize([m1, m2], score_threshold=0.5)
    assert s["true_positives"] == 1 and s["false_positives"] == 1 and s["false_negatives"] == 1
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    assert s["centre_error_px"]["median"] == pytest.approx(2 ** 0.5)
