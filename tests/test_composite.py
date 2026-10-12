import numpy as np

from ml.datasets.composite import Sprite, composite_crop, paste, render_ball


def _sprite(d=40):
    yy, xx = np.mgrid[0:d, 0:d]
    a = (((xx - d / 2) ** 2 + (yy - d / 2) ** 2) < (d / 2) ** 2).astype(np.float32)
    rgb = np.ones((d, d, 3), np.float32) * a[..., None]  # white ball, premultiplied
    return Sprite(np.dstack([rgb, a]), "synthetic")


def test_render_ball_size_and_blur():
    rng = np.random.default_rng(0)
    sharp = render_ball(_sprite(), 10, 0.0, 0.0, rng)
    blurred = render_ball(_sprite(), 10, 20.0, 0.0, rng)
    ys, xs = np.nonzero(sharp[..., 3] > 0.25)
    assert 8 <= xs.max() - xs.min() + 1 <= 12
    ys2, xs2 = np.nonzero(blurred[..., 3] > 0.05)
    assert (xs2.max() - xs2.min()) > (xs.max() - xs.min()) + 10  # streak along x


def test_paste_returns_box_where_pixels_changed():
    frame = np.full((200, 300, 3), 60, np.uint8)
    patch = render_ball(_sprite(), 12, 0.0, 0.0, np.random.default_rng(1))
    box = paste(frame, patch, 150, 100)
    x1, y1, x2, y2 = map(int, box)
    assert frame[y1:y2, x1:x2].mean() > 60 and frame[:50].mean() == 60
    assert paste(frame, patch, 2, 2) is None  # off-frame placement is rejected


def test_composite_crop_labels_and_negatives():
    rng = np.random.default_rng(2)
    frame = np.full((1080, 1920, 3), (40, 130, 50), np.uint8)
    img, boxes = composite_crop(frame, [_sprite()], rng, crop=960, n_balls=(2, 2))
    assert img.shape[:2] == (960, 960) and len(boxes) == 2
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= 960 and 0 <= y1 < y2 <= 960
    _, none = composite_crop(frame, [_sprite()], rng, negative=True)
    assert none == []
