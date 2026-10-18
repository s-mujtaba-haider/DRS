import numpy as np
import pytest

from ml.detection.base import FrameInfo, make_detection
from ml.tracking.kalman import KalmanCA2D
from ml.tracking.tracker import BallTracker


def _ball_path(n=30, fps=50.0):
    t = np.arange(n) / fps
    x = 200 + 600 * t
    y = 500 - 400 * t + 300 * t**2
    return t, x, y


def _dets(t, x, y, drop=(), noise=0.0, conf=0.8, seed=0):
    rng = np.random.default_rng(seed)
    frames, dets = [], []
    for i, (tt, xx, yy) in enumerate(zip(t, x, y)):
        frames.append((i, tt * 1000))
        if i in drop:
            continue
        cx, cy = xx + rng.normal(0, noise), yy + rng.normal(0, noise)
        dets.append(make_detection(FrameInfo(i, tt * 1000, "container_pts"), (cx - 4, cy - 4, cx + 4, cy + 4), conf, "test", "model"))
    return frames, dets


def test_kalman_initialisation():
    kf = KalmanCA2D.init(np.array([10.0, 20.0]), 3.0, 100.0, 100.0, 10.0)
    assert kf.x[:2].tolist() == [10.0, 20.0]
    assert np.all(kf.x[2:] == 0)
    assert kf.P[0, 0] == 9.0


def test_tracker_follows_clean_path():
    t, x, y = _ball_path()
    frames, dets = _dets(t, x, y, noise=0.5)
    res = BallTracker().run(frames, dets)
    assert res.primary_track_id is not None
    tr = res.tracks[0]
    assert tr.quality.num_observed == 30
    filt = [p for p in tr.points if p.kind == "filtered"]
    err = [np.hypot(p.x - x[p.frame_index], p.y - y[p.frame_index]) for p in filt]
    assert np.mean(err) < 1.0


def test_tracker_bridges_missing_frames_with_interpolated_points():
    t, x, y = _ball_path()
    frames, dets = _dets(t, x, y, drop={12, 13, 14})
    res = BallTracker().run(frames, dets)
    tr = res.tracks[0]
    interp = [p for p in tr.points if p.kind == "interpolated"]
    assert [p.frame_index for p in interp] == [12, 13, 14]
    assert tr.quality.longest_gap == 3
    for p in interp:
        assert abs(p.x - x[p.frame_index]) < 3 and abs(p.y - y[p.frame_index]) < 3
        assert p.sigma_x is not None and p.sigma_x > 0
    # Observations are preserved untouched.
    obs = [p for p in tr.points if p.kind == "observed"]
    assert all(p.x == pytest.approx(x[p.frame_index]) for p in obs)


def test_outlier_rejected():
    t, x, y = _ball_path()
    frames, dets = _dets(t, x, y)
    # Replace frame 15's detection by a far-away false positive.
    dets = [d for d in dets if d.frame_index != 15]
    dets.append(make_detection(FrameInfo(15, t[15] * 1000, "container_pts"), (900, 100, 908, 108), 0.9, "test", "model"))
    res = BallTracker().run(frames, dets)
    tr = res.tracks[0]
    assert all(not (p.frame_index == 15 and p.kind == "observed") for p in tr.points)


def test_static_clutter_ignored_and_ball_chosen():
    t, x, y = _ball_path()
    frames, dets = _dets(t, x, y, conf=0.5)
    for i, tt in enumerate(t):  # a stationary logo detected in every frame with high confidence
        dets.append(make_detection(FrameInfo(i, tt * 1000, "container_pts"), (50, 50, 60, 60), 0.95, "test", "model"))
    res = BallTracker().run(frames, dets)
    primary = res.tracks[0]
    assert primary.is_primary
    obs = [p for p in primary.points if p.kind == "observed"]
    assert obs[-1].x > 500  # it is the moving ball, not the logo


def test_no_detections():
    res = BallTracker().run([(0, 0.0), (1, 20.0)], [])
    assert res.tracks == [] and res.primary_track_id is None
    assert "No moving ball track found" in res.warnings


def test_no_association_across_shot_cut():
    t, x, y = _ball_path()
    frames, dets = _dets(t, x, y)
    res = BallTracker().run(frames, dets, shot_breaks={15})
    for tr in res.tracks:
        fr = [p.frame_index for p in tr.points if p.kind == "observed"]
        assert max(fr) < 15 or min(fr) >= 15, "track spans a camera cut"


def test_shot_cut_detector():
    from ml.video.shots import ShotCutDetector

    d = ShotCutDetector()
    green = np.zeros((90, 160, 3), np.uint8)
    green[:] = (40, 140, 40)
    crowd = np.random.default_rng(0).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    assert [d.is_cut(green), d.is_cut(green), d.is_cut(crowd), d.is_cut(crowd)] == [False, False, True, False]


def test_hysteresis_weak_detections_extend_but_do_not_spawn():
    from ml.tracking.tracker import TrackerConfig

    cfg = TrackerConfig(spawn_min_confidence=0.25, extend_min_confidence=0.05)
    t, x, y = _ball_path()
    frames, dets = _dets(t, x, y, conf=0.6)
    # Frames 10-20 have only weak detections of the ball: they should still extend the track.
    weak = [d.model_copy(update={"confidence": 0.1}) if 10 <= d.frame_index <= 20 else d for d in dets]
    res = BallTracker(cfg).run(frames, weak)
    obs = {p.frame_index for p in res.tracks[0].points if p.kind == "observed"}
    assert set(range(10, 21)) <= obs
    # Weak detections alone never create a track.
    _, only_weak = _dets(t, x, y, conf=0.1)
    assert BallTracker(cfg).run(frames, only_weak).tracks == []


def test_ghost_candidates_near_bounce_do_not_split_the_track():
    # Recorded motion-detector output for a synthetic delivery seen from behind the bowler: near the
    # bounce the ball barely moves in the image and ghost candidates appear beside it. A duplicate
    # hypothesis used to follow the same ball, steal the post-bounce detections, and leave the
    # primary track ending at the bounce (frames 33-35 lost), which broke the LBW projection.
    import json
    from pathlib import Path

    from ml.common.schemas import FrameDetections

    raw = json.loads((Path(__file__).parent / "data" / "ghost_bounce_detections.json").read_text())
    fds = [FrameDetections.model_validate(x) for x in raw]
    res = BallTracker().run([(f.frame_index, f.timestamp_ms) for f in fds], [d for f in fds for d in f.detections])
    primary = next(t for t in res.tracks if t.is_primary)
    obs = {p.frame_index for p in primary.points if p.kind == "observed"}
    assert {32, 33, 34, 35} <= obs, "post-bounce observations must stay on the ball track"
    assert len(res.tracks) == 1
