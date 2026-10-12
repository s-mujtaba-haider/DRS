"""Build a broadcast-domain ball training set: Kaggle balls (train split) composited onto real
broadcast frames at native resolution, merged with the Kaggle set itself.

    python scripts/build_broadcast_training_set.py --name ball_broadcast_v1

Split policy (no leakage):
* sprites only from data/processed/ball_kaggle/images/train
* background clips grouped by series (team pair); one series held out for validation
* series overlapping the independent broadcast test set (Zimbabwe matches) are excluded entirely
* Kaggle val/test splits are copied unchanged into val/test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets.composite import (  # noqa: E402
    composite_crop,
    load_sprites,
    sample_background_frames,
    summarize_rows,
    write_provenance,
    write_yolo_sample,
)
from ml.datasets.yolo_tools import validate, write_data_yaml  # noqa: E402


def series_of_clips(meta_dir: Path) -> dict[str, str]:
    out = {}
    for split in ("train", "val", "test"):
        for line in (meta_dir / f"{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            teams = sorted([r["response"]["batting_team"], r["response"]["bowling_team"]])
            out[Path(r["video"]).stem] = " v ".join(teams)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ball_broadcast_v1")
    ap.add_argument("--val-series", default="New Zealand v Pakistan")
    ap.add_argument("--exclude-team", default="Zimbabwe", help="exclude any series with this team (test-set overlap)")
    ap.add_argument("--every", type=int, default=12, help="sample every Nth frame from each clip")
    ap.add_argument("--per-video", type=int, default=12)
    ap.add_argument("--crops-per-frame", type=int, default=2)
    ap.add_argument("--negative-frac", type=float, default=0.15)
    ap.add_argument("--crop", type=int, default=960)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from ml.detection.yolo_detector import YoloBallDetector

    rng = np.random.default_rng(a.seed)
    kaggle = ROOT / "data" / "processed" / "ball_kaggle"
    clips_dir = ROOT / "data" / "raw" / "hf-cricket-delivery-videos"
    dst = ROOT / "data" / "processed" / a.name
    if dst.exists():
        shutil.rmtree(dst)

    sprites = load_sprites(kaggle, "train")
    print(f"sprites from Kaggle train split: {len(sprites)}")
    series = series_of_clips(clips_dir)
    videos = sorted((clips_dir / "videos").glob("*.mp4"))
    used = {s for v, s in series.items() if a.exclude_team not in s}
    print("series used:", sorted(used), "| validation series:", a.val_series)
    det = YoloBallDetector(ROOT / "models" / "checkpoints" / "cricket_ball_best.pt", conf=0.2, imgsz=1920, batch_size=1)

    rows: list[dict] = []
    stats = {"frames": 0, "boxes": 0}
    for v in videos:
        s = series.get(v.stem)
        if s is None or s not in used:
            continue
        split = "val" if s == a.val_series else "train"
        frames = sample_background_frames([v], a.every, a.per_video, ball_detector=det)
        for name, frame in frames:
            stats["frames"] += 1
            for k in range(a.crops_per_frame):
                neg = rng.random() < a.negative_frac
                img, boxes = composite_crop(frame, sprites, rng, crop=a.crop, negative=neg)
                stats["boxes"] += len(boxes)
                write_yolo_sample(dst, split, f"comp_{name}_{k}", img, boxes, rows, sequence=f"{s}|{v.stem}")
        print(f"\r{v.stem} {s:<28} frames={stats['frames']} boxes={stats['boxes']}", end="", flush=True)
    print()

    # Merge the Kaggle dataset (keeps its leak-free val/test splits).
    import csv

    kprov = {r["file"]: r for r in csv.DictReader((kaggle / "provenance.csv").open())}
    for split in ("train", "val", "test"):
        for img in sorted((kaggle / "images" / split).glob("*")):
            (dst / "images" / split).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst / "images" / split / img.name)
            lab = kaggle / "labels" / split / f"{img.stem}.txt"
            if lab.is_file():
                shutil.copy2(lab, dst / "labels" / split / lab.name)
            rows.append({"file": img.name, "source_sequence": "kaggle|" + kprov.get(img.name, {}).get("source_sequence", img.stem),
                         "label_origin": "human", "split": split})
    write_provenance(dst, rows)
    write_data_yaml(dst, ["ball"])
    rep = validate(dst, 1, check_duplicates=False)
    print(json.dumps({"summary": summarize_rows(rows), **stats, "validation_ok": rep.ok, "per_split": rep.per_split,
                      "box_size_px": rep.box_size_px, "split_leaks": rep.split_leaks}, indent=1))
    return 0 if rep.ok else 2


if __name__ == "__main__":
    sys.exit(main())
