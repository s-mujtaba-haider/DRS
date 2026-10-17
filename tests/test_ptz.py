import cv2
import numpy as np

from ml.video.ptz import estimate_shot_motion, stabilise_observations


def _scene(w=1280, h=720, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h * 2, w * 2, 3), (60, 130, 70), np.uint8)
    for _ in range(900):  # textured "crowd / pitch" with plenty of corners
        x, y = rng.integers(0, w * 2), rng.integers(0, h * 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(4, 30)), int(y + rng.integers(4, 30))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def _zoom_pan(k, w=1280, h=720):
    """Homography from the big scene into frame k: zoom in 1% per frame plus a slow pan."""
    s = 0.5 * (1.01 ** k)
    tx, ty = -0.25 * w * 2 * s + w / 2 - 2.0 * k, -0.25 * h * 2 * s + h / 2 - 1.0 * k
    return np.array([[s, 0, tx + (1 - s) * 0], [0, s, ty], [0, 0, 1]], float)


def _frames(n=20, w=1280, h=720):
    scene = _scene(w, h)
    out = {}
    for k in range(n):
        f = cv2.warpPerspective(scene, _zoom_pan(k), (w, h))
        cv2.rectangle(f, (0, h - 60), (w, h), (30, 30, 30), -1)  # static score bar
        cv2.putText(f, "SCORE 123-4  OVERS 12.3", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out[k] = f
    return out


def _true_map(k, ref, p):
    """Map a point from frame k to frame ref with the generating homographies."""
    H = _zoom_pan(ref) @ np.linalg.inv(_zoom_pan(k))
    q = H @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def test_zoom_and_pan_are_compensated_despite_static_overlay():
    frames = _frames()
    ref = 10
    motion = estimate_shot_motion(frames, ref)
    assert set(motion.to_ref) == set(frames)
    for k in (0, 5, 15, 19):
        for p in ((300.0, 200.0), (900.0, 500.0)):
            est = motion.map_points(k, np.array([p]))[0]
            assert np.linalg.norm(est - _true_map(k, ref, p)) < 1.5
    # Frame 0 precedes ten 1% zoom steps: it is zoomed out relative to the reference.
    assert abs(motion.zoom(0) - 1.01 ** -10) < 0.01


def test_stabilise_observations_maps_points_and_radii(tmp_path):
    frames = _frames(16)
    path = tmp_path / "zoom.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
    for k in sorted(frames):
        vw.write(frames[k])
    vw.release()
    fr = np.array([2, 6, 12])
    uv = np.array([[400.0, 300.0], [700.0, 350.0], [640.0, 360.0]])
    rad = np.array([5.0, 5.0, 5.0])
    st = stabilise_observations(str(path), fr, uv, rad, ref_frame=8)
    assert st.keep.all()
    for k, p, q in zip(fr, uv, st.uv):
        assert np.linalg.norm(q - _true_map(int(k), 8, p)) < 2.5  # mp4 compression adds a little error
    # Frame 2 is zoomed out relative to frame 8, so the ball looks bigger once mapped into frame 8.
    assert st.radius[0] > 5.0 > st.radius[2]
    assert "compensated" in st.notes[0]
