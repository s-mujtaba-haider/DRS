"""YOLO-format dataset utilities: validation, statistics, duplicates, leak-free splits, visualisation.

Layout handled:  <root>/images/<split>/*.jpg  and  <root>/labels/<split>/*.txt
Label lines:     class cx cy w h   (normalised to [0, 1])

Provenance is tracked per sample in ``provenance.csv`` (file, source_sequence,
label_origin) where label_origin is one of human | auto | auto_corrected |
synthetic_exact | unlabeled. Splits are made by ``source_sequence`` (video /
clip / match) so neighbouring frames never land in different splits.
"""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DRS_CLASSES = ["ball", "bat", "stumps", "bails", "batter", "bowler", "wicketkeeper", "fielder", "umpire", "pad", "crease"]
LABEL_ORIGINS = {"human", "auto", "auto_corrected", "synthetic_exact", "unlabeled"}


@dataclass
class ValidationReport:
    root: str
    images: int = 0
    labels: int = 0
    boxes: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    invalid_lines: list[str] = field(default_factory=list)
    unknown_classes: Counter = field(default_factory=Counter)
    class_counts: Counter = field(default_factory=Counter)
    per_split: dict = field(default_factory=dict)
    box_size_px: dict = field(default_factory=dict)
    duplicates: list[list[str]] = field(default_factory=list)
    split_leaks: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.invalid_lines or self.unknown_classes or self.split_leaks)

    def as_dict(self) -> dict:
        d = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in self.__dict__.items()}
        d["ok"] = self.ok
        for k in ("missing_labels", "orphan_labels", "invalid_lines"):
            d[f"num_{k}"] = len(d[k])
            d[k] = d[k][:50]
        return d


def dhash(path: Path, size: int = 8) -> str | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return hashlib.sha1(np.packbits(bits).tobytes()).hexdigest()


def read_provenance(root: Path) -> dict[str, dict]:
    p = root / "provenance.csv"
    if not p.is_file():
        return {}
    with p.open() as fh:
        return {r["file"]: r for r in csv.DictReader(fh)}


def validate(root: str | Path, num_classes: int, check_duplicates: bool = True) -> ValidationReport:
    root = Path(root)
    rep = ValidationReport(root=str(root))
    sizes: list[float] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    prov = read_provenance(root)
    seq_splits: dict[str, set[str]] = defaultdict(set)
    for split_dir in sorted((root / "images").iterdir()) if (root / "images").is_dir() else []:
        split = split_dir.name
        n_split = 0
        for img in sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMG_EXT):
            rep.images += 1
            n_split += 1
            lab = root / "labels" / split / img.relative_to(split_dir).with_suffix(".txt")
            if prov.get(img.name, {}).get("source_sequence"):
                seq_splits[prov[img.name]["source_sequence"]].add(split)
            if check_duplicates:
                h = dhash(img)
                if h:
                    hashes[h].append(f"{split}/{img.name}")
            if not lab.is_file():
                rep.missing_labels.append(f"{split}/{img.name}")  # allowed: negative image
                continue
            rep.labels += 1
            im = cv2.imread(str(img))
            H, W = im.shape[:2] if im is not None else (0, 0)
            for ln, line in enumerate(lab.read_text().splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    c = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    if len(parts) != 5 or not all(0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                        raise ValueError
                except ValueError:
                    rep.invalid_lines.append(f"{lab.name}:{ln}: {line[:60]}")
                    continue
                if c >= num_classes or c < 0:
                    rep.unknown_classes[c] += 1
                    continue
                rep.boxes += 1
                rep.class_counts[c] += 1
                if W:
                    sizes.append(float(np.sqrt(w * W * h * H)))
        rep.per_split[split] = n_split
    rep.orphan_labels = [str(p.relative_to(root)) for p in (root / "labels").rglob("*.txt")
                         if not any((root / "images" / p.relative_to(root / "labels")).with_suffix(e).is_file() for e in IMG_EXT)] \
        if (root / "labels").is_dir() else []
    rep.duplicates = [v for v in hashes.values() if len(v) > 1][:100]
    rep.split_leaks = sorted(s for s, splits in seq_splits.items() if len(splits) > 1)
    if sizes:
        a = np.array(sizes)
        rep.box_size_px = {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
                           "frac_tiny_lt_16px": float(np.mean(a < 16)), "frac_small_lt_32px": float(np.mean(a < 32))}
    return rep


def split_by_sequence(sequences: dict[str, list[str]], ratios=(0.7, 0.15, 0.15), seed: int = 0) -> dict[str, str]:
    """Assign whole sequences to train/val/test. Returns item -> split."""
    keys = sorted(sequences)
    random.Random(seed).shuffle(keys)
    total = sum(len(sequences[k]) for k in keys)
    seq_split: dict[str, str] = {}
    acc = 0
    for k in keys:
        frac = acc / max(total, 1)
        seq_split[k] = "train" if frac < ratios[0] else "val" if frac < ratios[0] + ratios[1] else "test"
        acc += len(sequences[k])
    # With few sequences, guarantee non-empty val/test by moving whole sequences (never frames).
    for needed in ("test", "val"):
        if needed not in seq_split.values():
            donors = [k for k in reversed(keys) if seq_split[k] == "train"]
            if len(donors) > 1:
                seq_split[donors[0]] = needed
    return {item: seq_split[k] for k in keys for item in sequences[k]}


def draw_labels(img_path: Path, label_path: Path, names: list[str]) -> np.ndarray:
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            c, cx, cy, w, h = line.split()[:5]
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            cv2.rectangle(img, (x1, y1), (int(cx + w / 2), int(cy + h / 2)), (0, 255, 0), 2)
            cv2.putText(img, names[int(c)] if int(c) < len(names) else c, (x1, max(12, y1 - 4)), 0, 0.5, (0, 255, 0), 1)
    return img


def write_data_yaml(root: Path, names: list[str]) -> Path:
    import yaml

    p = root / "data.yaml"
    p.write_text(yaml.safe_dump({"path": str(root.resolve()).replace("\\", "/"), "train": "images/train",
                                 "val": "images/val", "test": "images/test", "names": dict(enumerate(names))}, sort_keys=False))
    return p


def convert_cricket_synth(src: Path, dst: Path, include_negatives: bool = True, seed: int = 0) -> dict:
    """cricket-synth (exact synthetic labels) -> YOLO 'ball' dataset, split by clip."""
    import json

    labels_dir, clips_dir = src / "out" / "labels", src / "out" / "clips"
    seqs: dict[str, list[tuple[Path, list | None]]] = {}
    for clip in sorted(clips_dir.iterdir()) if clips_dir.is_dir() else []:
        lab = labels_dir / f"{clip.name}.json"
        if not lab.is_file():
            continue
        d = json.loads(lab.read_text())
        by_frame = {f["frame"]: f for f in d["frames"]}
        items = []
        for img in sorted(clip.glob("*.jpg")):
            fr = by_frame.get(int(img.stem.split("_")[-1]))
            if fr is None:
                continue
            box = fr["bbox_px"] if fr.get("visible") and fr.get("bbox_px") else None
            if box is None and not include_negatives:
                continue
            items.append((img, box))
        if items:
            seqs[clip.name] = items
    assign = split_by_sequence({k: [str(i[0]) for i in v] for k, v in seqs.items()}, seed=seed)
    if dst.exists():
        shutil.rmtree(dst)
    rows = []
    for clip, items in seqs.items():
        for img, box in items:
            split = assign[str(img)]
            name = f"{clip}_{img.name}"
            (dst / "images" / split).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst / "images" / split / name)
            H, W = cv2.imread(str(img)).shape[:2]
            lines = []
            if box:
                x, y, w, h = box
                lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            (dst / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines))
            rows.append({"file": name, "source_sequence": clip, "label_origin": "synthetic_exact", "split": split})
    with (dst / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source_sequence", "label_origin", "split"])
        w.writeheader()
        w.writerows(rows)
    write_data_yaml(dst, ["ball"])
    return {"clips": len(seqs), "images": len(rows), "splits": dict(Counter(r["split"] for r in rows))}
