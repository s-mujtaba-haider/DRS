"""Crease mark proposals: batter selection, grounded hints and video plumbing with a fake pose model."""

from __future__ import annotations

import numpy as np
import pytest

from ml.runout.proposals import (
    L_ANKLE,
    L_KNEE,
    R_ANKLE,
    R_KNEE,
    PersonPose,
    propose_from_poses,
    propose_in_video,
    select_batter,
)


def person(cx: float, foot_y: float, h: float = 200.0, l_dy: float = 0.0, r_dy: float = 0.0, conf: float = 0.9,
           score: float = 0.9) -> PersonPose:
    """Standing person; l_dy / r_dy raise a foot (negative = up)."""
    kp = np.zeros((17, 3))
    kp[:, 2] = conf
    kp[:, 0] = cx
    kp[:, 1] = foot_y - h / 2
    kp[L_ANKLE] = (cx - 10, foot_y + l_dy, conf)
    kp[R_ANKLE] = (cx + 10, foot_y + r_dy, conf)
    kp[L_KNEE] = (cx - 10, foot_y + l_dy - h * 0.25, conf)
    kp[R_KNEE] = (cx + 10, foot_y + r_dy - h * 0.25, conf)
    return PersonPose((cx - 30, foot_y - h, cx + 30, foot_y + 2), score, kp)


CREASE = ((0.0, 500.0), (1000.0, 500.0))


def test_select_batter_nearest_crease_and_roi():
    near, far = person(300, 505), person(700, 300)
    p, why = select_batter([far, near], crease=CREASE)
    assert p is near and why == "nearest the crease"
    p, _ = select_batter([near, far], roi=(600, 100, 800, 400))
    assert p is far
    p, why = select_batter([near], roi=(600, 100, 800, 400))
    assert p is None and "region" in why
    assert select_batter([], crease=CREASE)[0] is None


def test_select_batter_prefers_continuity():
    a, b = person(300, 505), person(330, 505)
    p, _ = select_batter([a, b], crease=CREASE, previous=(335, 505))
    assert p is b


def test_grounded_hint_still_lower_foot_true_raised_foot_false():
    # left foot planted at y=505; right foot swings up and forward
    seq = [[person(300, 505, r_dy=-d)] for d in (0, 30, 60, 90)]
    res = propose_from_poses([(i, i * 20.0) for i in range(4)], seq, "fake", crease=CREASE)
    f1 = {p.kind: p for p in res.frames[1].proposals if p.kind.startswith("ankle")}
    assert f1["ankle_left"].grounded_hint is True
    assert f1["ankle_right"].grounded_hint is False
    # end frames lack a neighbour on one side: no hint rather than a guess
    assert all(p.grounded_hint is None for p in res.frames[0].proposals)
    assert all(p.source == "model" and "Confirm" in p.note for f in res.frames for p in f.proposals)


def test_low_confidence_and_toe_estimate():
    seq = [[person(300, 505, conf=0.2)]] * 3
    res = propose_from_poses([(i, i * 20.0) for i in range(3)], seq, "fake", crease=CREASE)
    assert res.frames_with_batter == 3 and res.frames[1].proposals == []  # ankles below 0.3 not proposed
    seq = [[person(300, 505)]] * 3
    res = propose_from_poses([(i, i * 20.0) for i in range(3)], seq, "fake", crease=CREASE)
    toe = next(p for p in res.frames[1].proposals if p.kind == "toe_est")
    assert toe.point[1] > 505  # below the ankle, along the shin direction
    assert any("Bat tip is not proposed" in w for w in res.warnings)


def test_missing_batter_frames_are_reported():
    seq = [[person(300, 505)], [], [person(300, 505)]]
    res = propose_from_poses([(i, i * 20.0) for i in range(3)], seq, "fake", crease=CREASE)
    assert res.frames_with_batter == 2 and not res.frames[1].batter_found
    assert any("not found in 1 of 3" in w for w in res.warnings)
    res = propose_from_poses([(0, 0.0)], [[person(300, 505)]], "fake")
    assert res.frames_with_batter == 0 and any("cannot be identified" in w for w in res.warnings)


class FakePose:
    model_id = "fake-pose"

    def __init__(self):
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        return [[person(160, 200, h=120)] for _ in images]


def test_propose_in_video(tmp_path):
    from ml.video.ingest import write_video

    path = write_video((np.full((240, 320, 3), 60, np.uint8) for _ in range(20)), tmp_path / "v.mp4", 50)
    m = FakePose()
    res = propose_in_video(path, m, 4, 14, crease=((0, 200), (320, 200)), batch=4)
    assert [f.frame_index for f in res.frames] == list(range(4, 14)) and m.calls == 3
    assert res.frames[0].timestamp_ms == pytest.approx(80.0, abs=1)
    assert res.inference_fps is not None
    with pytest.raises(ValueError):
        propose_in_video(path, m, 5, 5)
    with pytest.raises(ValueError):
        propose_in_video(path, m, 0, 1000)
