"""Broadcast-domain small-ball training data by copy-paste compositing.

Real cricket-ball appearances (cropped from a labelled *training* split) are shrunk to
broadcast scale, motion-blurred along a random direction, colour/brightness-matched to the
local background and pasted onto real broadcast frames that contain no detectable ball.
Labels are exact by construction (``label_origin = synthetic_composite``).

This addresses the measured domain gap (training balls ~66 px, broadcast balls ~7-13 px). It is
not a substitute for real labelled broadcast frames: compositing cannot reproduce occlusion by
players, true lighting interaction or the real motion-blur profile exactly. Always evaluate on
real, independent frames.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ml.datasets.yolo_tools import IMG_EXT


@dataclass
class Sprite:
    rgba: np.ndarray  # float32 HxWx4, colour in 0..1, premultiplied alpha
    source: str


def load_sprites(yolo_root: Path, split: str = "train", min_size: int = 20, max_aspect: float = 1.35,
                 limit: int | None = None, seed: int = 0) -> list[Sprite]:
    """Crop balls from a YOLO dataset split, with a feathered elliptical alpha (drops bbox corners)."""
    rng = random.Random(seed)
    imgs = sorted(p for p in (yolo_root / "images" / split).glob("*") if p.suffix.lower() in IMG_EXT)
    rng.shuffle(imgs)
    out: list[Sprite] = []
    for p in imgs:
        lab = yolo_root / "labels" / split / f"{p.stem}.txt"
        if not lab.is_file():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        H, W = im.shape[:2]
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:5])
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < min_size or max(bw, bh) / max(1, min(bw, bh)) > max_aspect:
                continue
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue
            crop = im[y1:y2, x1:x2].astype(np.float32) / 255.0
            yy, xx = np.mgrid[0:bh, 0:bw]
            r = np.sqrt(((xx + 0.5 - bw / 2) / (bw / 2)) ** 2 + ((yy + 0.5 - bh / 2) / (bh / 2)) ** 2)
            alpha = np.clip((1.0 - r) / 0.12, 0, 1).astype(np.float32)  # soft edge
            rgba = np.dstack([crop[..., ::-1] * alpha[..., None], alpha])  # RGB premultiplied
            out.append(Sprite(rgba, p.name))
            if limit and len(out) >= limit:
                return out
    return out


def _motion_kernel(length: float, angle: float) -> np.ndarray:
    n = max(1, int(math.ceil(length)))
    k = np.zeros((n * 2 + 1, n * 2 + 1), np.float32)
    c = n
    dx, dy = math.cos(angle), math.sin(angle)
    for t in np.linspace(-length / 2, length / 2, max(2, int(length * 4))):
        k[int(round(c + t * dy)), int(round(c + t * dx))] = 1
    return k / k.sum()


def render_ball(sprite: Sprite, diameter: float, blur_len: float, angle: float, rng: np.random.Generator) -> np.ndarray:
    """Return premultiplied RGBA float patch for a ball of ``diameter`` px with motion blur."""
    s = sprite.rgba
    scale = diameter / max(s.shape[:2])
    w = max(2, int(round(s.shape[1] * scale)))
    h = max(2, int(round(s.shape[0] * scale)))
    small = cv2.resize(s, (w, h), interpolation=cv2.INTER_AREA)
    # colour / brightness jitter applied to the premultiplied colour
    a = small[..., 3:4]
    col = np.divide(small[..., :3], np.maximum(a, 1e-6))
    col = np.clip(col * rng.uniform(0.75, 1.3) + rng.normal(0, 0.03, 3), 0, 1)
    sat = rng.uniform(0.5, 1.0)  # broadcast balls usually look desaturated
    grey = col.mean(axis=2, keepdims=True)
    col = grey + (col - grey) * sat
    small = np.dstack([col * a, a])
    pad = int(math.ceil(blur_len)) + 2
    small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    if blur_len >= 1.0:
        small = cv2.filter2D(small, -1, _motion_kernel(blur_len, angle))
    return small


def paste(frame: np.ndarray, patch: np.ndarray, cx: float, cy: float) -> tuple[float, float, float, float] | None:
    """Alpha-composite premultiplied ``patch`` (RGB order) onto BGR uint8 ``frame``; returns xyxy box."""
    H, W = frame.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
    if x0 < 0 or y0 < 0 or x0 + pw > W or y0 + ph > H:
        return None
    roi = frame[y0:y0 + ph, x0:x0 + pw].astype(np.float32) / 255.0
    rgb = patch[..., 2::-1]  # RGB -> BGR
    a = patch[..., 3:4]
    out = rgb + roi * (1 - a)
    frame[y0:y0 + ph, x0:x0 + pw] = np.clip(out * 255, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(patch[..., 3] > 0.25)
    if len(xs) == 0:
        return None
    return float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1), float(y0 + ys.max() + 1)


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild broadcast-like degradation: slight blur, sensor noise, JPEG re-compression."""
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.8))
    noise = rng.normal(0, rng.uniform(0, 3.0), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    q = int(rng.integers(55, 92))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def composite_crop(frame: np.ndarray, sprites: list[Sprite], rng: np.random.Generator, crop: int = 960,
                   n_balls: tuple[int, int] = (1, 3), diam_range: tuple[float, float] = (5.0, 22.0),
                   negative: bool = False) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Native-resolution crop of ``frame`` with 0..n pasted balls. Returns (image, xyxy boxes)."""
    H, W = frame.shape[:2]
    cw, ch = min(crop, W), min(crop, H)
    x0 = int(rng.integers(0, W - cw + 1))
    y0 = int(rng.integers(0, H - ch + 1))
    img = frame[y0:y0 + ch, x0:x0 + cw].copy()
    boxes: list[tuple[float, float, float, float]] = []
    if not negative:
        for _ in range(int(rng.integers(n_balls[0], n_balls[1] + 1))):
            d = float(np.exp(rng.uniform(np.log(diam_range[0]), np.log(diam_range[1]))))
            blur = float(rng.uniform(0, 2.5 * d)) if rng.random() < 0.7 else 0.0
            patch = render_ball(sprites[int(rng.integers(len(sprites)))], d, blur, float(rng.uniform(0, np.pi)), rng)
            for _try in range(10):
                cx = float(rng.uniform(patch.shape[1], cw - patch.shape[1]))
                cy = float(rng.uniform(patch.shape[0], ch - patch.shape[0]))
                if all(abs(cx - (b[0] + b[2]) / 2) > 40 or abs(cy - (b[1] + b[3]) / 2) > 40 for b in boxes):
                    box = paste(img, patch, cx, cy)
                    if box:
                        boxes.append(box)
                    break
    return degrade(img, rng), boxes


def sample_background_frames(videos: list[Path], every: int, max_per_video: int, ball_detector=None,
                             skip_conf: float = 0.3) -> list[tuple[str, np.ndarray]]:
    """Frames from broadcast clips; frames where ``ball_detector`` already sees a ball are skipped
    (an unlabeled real ball would teach the model that balls are background)."""
    from ml.detection.base import FrameInfo
    from ml.video.ingest import iter_frames

    out = []
    for v in videos:
        n = 0
        for rec in iter_frames(v, step=every):
            if ball_detector is not None:
                dets = ball_detector.detect_batch([rec.image], [FrameInfo(rec.index, rec.timestamp_ms, rec.timestamp_source)])[0]
                if any(d.confidence >= skip_conf for d in dets):
                    continue
            out.append((f"{v.stem}_f{rec.index:05d}", rec.image))
            n += 1
            if n >= max_per_video:
                break
    return out


def write_yolo_sample(dst: Path, split: str, name: str, img: np.ndarray, boxes, rows: list, sequence: str) -> None:
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    H, W = img.shape[:2]
    cv2.imwrite(str(dst / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = [f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}" for b in boxes]
    (dst / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    rows.append({"file": f"{name}.jpg", "source_sequence": sequence, "label_origin": "synthetic_composite", "split": split})


def summarize_rows(rows: list[dict]) -> dict:
    return {"images": len(rows), "splits": dict(Counter(r["split"] for r in rows))}


def write_provenance(dst: Path, rows: list[dict]) -> None:
    with (dst / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source_sequence", "label_origin", "split"])
        w.writeheader()
        w.writerows(rows)
