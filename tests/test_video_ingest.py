import numpy as np
import pytest

from ml.common.schemas import TimestampSource
from ml.video.ingest import VideoDecodeError, extract_frames, iter_frames, probe_video, write_video


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"

    def frames():
        for i in range(30):
            img = np.zeros((96, 128, 3), np.uint8)
            img[:, : 4 * i % 128] = 200
            yield img

    write_video(frames(), path, 25)
    return path


def test_probe_metadata(tiny_video):
    meta = probe_video(tiny_video, full_scan=True)
    assert (meta.width, meta.height) == (128, 96)
    assert meta.nominal_fps == pytest.approx(25, rel=1e-3)
    assert meta.frame_count_decoded == 30
    assert meta.variable_frame_rate is False
    assert meta.pts_delta_ms_mean == pytest.approx(40, abs=0.5)


def test_iter_frames_timestamps_and_step(tiny_video):
    recs = list(iter_frames(tiny_video, step=3))
    assert [r.index for r in recs] == list(range(0, 30, 3))
    assert recs[0].timestamp_ms == pytest.approx(0.0)
    assert recs[1].timestamp_ms == pytest.approx(120.0, abs=1)
    assert all(r.timestamp_source == TimestampSource.CONTAINER_PTS for r in recs)
    assert recs[0].image.shape == (96, 128, 3)


def test_iter_frames_window_and_max(tiny_video):
    recs = list(iter_frames(tiny_video, start_frame=5, end_frame=20, max_frames=4))
    assert [r.index for r in recs] == [5, 6, 7, 8]


def test_extract_frames_writes_index(tiny_video, tmp_path):
    idx = extract_frames(tiny_video, tmp_path / "frames", step=10)
    lines = idx.read_text().strip().splitlines()
    assert lines[0].startswith("file,frame_index,timestamp_ms")
    assert len(lines) == 4
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 3


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    with pytest.raises(VideoDecodeError, match="empty"):
        probe_video(p)


def test_invalid_file_rejected(tmp_path):
    p = tmp_path / "junk.mp4"
    p.write_bytes(b"this is not a video" * 100)
    with pytest.raises(VideoDecodeError):
        probe_video(p)


def test_missing_file(tmp_path):
    with pytest.raises(VideoDecodeError, match="not found"):
        probe_video(tmp_path / "nope.mp4")


def test_invalid_step(tiny_video):
    with pytest.raises(ValueError):
        list(iter_frames(tiny_video, step=0))
