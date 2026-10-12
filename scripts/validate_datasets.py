"""Validate the dataset registry and any prepared YOLO datasets.

    python scripts/validate_datasets.py                 # registry + all data/processed/* YOLO sets
    python scripts/validate_datasets.py --viz 12        # also write annotation previews

Checks: registry fields, local presence of downloaded data, YOLO label validity, class
distribution, box sizes, near-duplicate images (dHash) and train/val/test sequence leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets.yolo_tools import IMG_EXT, draw_labels, validate  # noqa: E402

REQUIRED = ["id", "name", "source", "url", "download_method", "license", "commercial_use", "has_ball_annotations",
            "has_trajectories_or_3d", "multi_view", "auth_required", "intended_use", "status"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", type=int, default=0, help="write N annotated previews per dataset")
    a = ap.parse_args()
    problems = 0
    reg = json.loads((ROOT / "data" / "dataset_registry.json").read_text())
    for d in reg["datasets"]:
        missing = [k for k in REQUIRED if k not in d]
        local = ROOT / d["local_path"] if d.get("local_path") else None
        present = bool(local and local.exists() and any(local.iterdir()))
        flag = "OK"
        if missing:
            flag, problems = f"MISSING FIELDS {missing}", problems + 1
        elif d["status"] == "downloaded" and not present:
            flag, problems = "registry says downloaded but data is absent (run scripts/download_datasets.py)", problems + 1
        print(f"[registry] {d['id']:<38} {d['status']:<34} {flag}")
    import yaml

    for data_yaml in sorted((ROOT / "data" / "processed").glob("*/data.yaml")):
        root = data_yaml.parent
        y = yaml.safe_load(data_yaml.read_text())
        names = list(y["names"].values()) if isinstance(y["names"], dict) else list(y["names"])
        rep = validate(root, len(names))
        out = rep.as_dict()
        print(f"[yolo] {root.name}: ok={rep.ok} images={rep.images} boxes={rep.boxes} classes={dict(rep.class_counts)} "
              f"splits={rep.per_split} sizes={rep.box_size_px} duplicates={len(rep.duplicates)} leaks={rep.split_leaks}")
        (root / "validation_report.json").write_text(json.dumps(out, indent=2, default=str))
        problems += 0 if rep.ok else 1
        if a.viz:
            vdir = root / "previews"
            vdir.mkdir(exist_ok=True)
            imgs = [p for p in (root / "images").rglob("*") if p.suffix.lower() in IMG_EXT][: a.viz]
            for p in imgs:
                split = p.parent.name
                cv2.imwrite(str(vdir / p.name), draw_labels(p, root / "labels" / split / f"{p.stem}.txt", names))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
