"""Merge prepared YOLO datasets (same class list), keeping each sample's split and provenance.

    python scripts/merge_yolo_datasets.py --sources ball_kaggle ball_pseudo_v1 --name ball_mix_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets.yolo_tools import IMG_EXT, validate, write_data_yaml  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="names under data/processed")
    ap.add_argument("--name", required=True)
    ap.add_argument("--names", nargs="+", default=["ball"])
    a = ap.parse_args()
    dst = ROOT / "data" / "processed" / a.name
    if dst.exists():
        shutil.rmtree(dst)
    rows = []
    for src_name in a.sources:
        src = ROOT / "data" / "processed" / src_name
        prov = {r["file"]: r for r in csv.DictReader((src / "provenance.csv").open())} if (src / "provenance.csv").is_file() else {}
        for img in sorted(p for p in (src / "images").rglob("*") if p.suffix.lower() in IMG_EXT):
            split = img.parent.name
            (dst / "images" / split).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
            name = f"{src_name}__{img.name}"
            shutil.copy2(img, dst / "images" / split / name)
            lab = src / "labels" / split / f"{img.stem}.txt"
            if lab.is_file():
                shutil.copy2(lab, dst / "labels" / split / f"{Path(name).stem}.txt")
            p = prov.get(img.name, {})
            rows.append({"file": name, "source_sequence": f"{src_name}|{p.get('source_sequence', img.stem)}",
                         "label_origin": p.get("label_origin", "unknown"), "split": split})
    with (dst / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source_sequence", "label_origin", "split"])
        w.writeheader()
        w.writerows(rows)
    write_data_yaml(dst, a.names)
    rep = validate(dst, len(a.names), check_duplicates=False)
    print(json.dumps({"images": rep.images, "boxes": rep.boxes, "per_split": rep.per_split, "ok": rep.ok,
                      "box_size_px": rep.box_size_px, "split_leaks": rep.split_leaks[:5]}, indent=1))
    return 0 if rep.ok else 2


if __name__ == "__main__":
    sys.exit(main())
