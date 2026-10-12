"""Convert SuperAnnotate pixel annotations (e.g. Acme AI "Cricket Semantic Segmentation") to YOLO boxes.

Each ``<image>___pixel.json`` lists instances with ``parts[].color``; ``<image>___save.png``
encodes every part as that exact RGB colour. A box is the bounding rectangle of all pixels of an
instance's parts. Pixel-exact colour matching avoids anti-aliasing ambiguity (SuperAnnotate
save masks are not blended).
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ml.datasets.yolo_tools import write_data_yaml

# Source class name -> DRS class name (unlisted classes are skipped).
CLASS_MAP = {"Ball": "ball", "Wicket": "stumps", "Batsmen": "batter", "Bowler": "bowler",
             "Wicket Keeper": "wicketkeeper", "Umpire": "umpire", "Fielder": "fielder"}
NAMES = ["ball", "stumps", "batter", "bowler", "wicketkeeper", "umpire", "fielder"]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def instance_boxes(pixel_json: Path, save_png: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    meta = json.loads(pixel_json.read_text())
    mask = cv2.imread(str(save_png), cv2.IMREAD_COLOR)
    if mask is None:
        return []
    rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    # Pack colours into one int per pixel for fast equality tests.
    packed = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)
    out = []
    for inst in meta.get("instances", []):
        name = CLASS_MAP.get(inst.get("className"))
        if not name:
            continue
        sel = np.zeros(packed.shape, bool)
        for part in inst.get("parts", []):
            r, g, b = _hex_to_rgb(part["color"])
            sel |= packed == ((r << 16) | (g << 8) | b)
        ys, xs = np.nonzero(sel)
        if len(xs) == 0:
            continue
        out.append((name, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)))
    return out


def convert(src_images: Path, dst: Path, split: str = "test", source_name: str = "acmeai_cricket_seg") -> dict:
    """All images go to one split: the set is used as an independent test domain, not for training."""
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images" / split).mkdir(parents=True)
    (dst / "labels" / split).mkdir(parents=True)
    counts: Counter = Counter()
    rows = []
    for pj in sorted(src_images.glob("*___pixel.json")):
        img_name = pj.name.removesuffix("___pixel.json")
        img = src_images / img_name
        save = src_images / f"{img_name}___save.png"
        if not img.is_file() or not save.is_file():
            continue
        im = cv2.imread(str(img))
        H, W = im.shape[:2]
        safe = img_name.replace(" ", "_").replace("(", "").replace(")", "")
        stem = Path(safe).stem
        cv2.imwrite(str(dst / "images" / split / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for name, (x1, y1, x2, y2) in instance_boxes(pj, save):
            c = NAMES.index(name)
            counts[name] += 1
            lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        (dst / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        rows.append({"file": f"{stem}.jpg", "source_sequence": source_name, "label_origin": "human", "split": split})
    with (dst / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source_sequence", "label_origin", "split"])
        w.writeheader()
        w.writerows(rows)
    write_data_yaml(dst, NAMES)
    return {"images": len(rows), "boxes_per_class": dict(counts)}


# Frame-number blocks (match time order) for the stumps-only derivative: the first innings trains,
# the later second innings tests, so neighbouring near-identical frames never straddle splits.
STUMPS_SPLITS = (("train", 0, 175), ("val", 176, 260), ("test", 261, 10**9))


def derive_stumps_set(seg_test: Path, dst: Path, splits=STUMPS_SPLITS) -> dict:
    """broadcast_seg_test (all classes, test-only) -> single-class 'stumps' set split by match time.

    The ball detector never trains on these frames; only the separate stumps model does.
    """
    import re

    if dst.exists():
        shutil.rmtree(dst)
    stumps_id = NAMES.index("stumps")
    rows, counts = [], Counter()
    for img in sorted((seg_test / "images" / "test").glob("*.jpg")):
        n = int(re.search(r"_(\d+)", img.stem).group(1))
        split = next(s for s, lo, hi in splits if lo <= n <= hi)
        lab = seg_test / "labels" / "test" / f"{img.stem}.txt"
        lines = [" ".join(["0", *p[1:]]) for p in (ln.split() for ln in lab.read_text().splitlines())
                 if p and int(p[0]) == stumps_id]
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst / "images" / split / img.name)
        (dst / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
        counts[split] += len(lines)
        rows.append({"file": img.name, "source_sequence": f"acmeai_frame_{n}", "label_origin": "human", "split": split})
    with (dst / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source_sequence", "label_origin", "split"])
        w.writeheader()
        w.writerows(rows)
    write_data_yaml(dst, ["stumps"])
    return {"images": len(rows), "boxes_per_split": dict(counts)}
