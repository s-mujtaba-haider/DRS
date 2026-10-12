"""Convert downloaded datasets into YOLO training sets (split by source sequence).

    python scripts/prepare_datasets.py synth           # cricket-synth -> data/processed/synth_yolo
    python scripts/prepare_datasets.py broadcast-seg   # Kaggle seg masks -> data/processed/broadcast_seg_test (test only)
    python scripts/prepare_datasets.py stumps-seg      # its stumps boxes -> data/processed/stumps_seg (split by match time)
    python scripts/prepare_datasets.py yolo-dir PATH   # adopt an existing YOLO export (e.g. Kaggle/Roboflow)
                                                       #  -> data/processed/<name>, provenance + leak check
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

from ml.datasets.yolo_tools import IMG_EXT, convert_cricket_synth, rebuild_leak_free, validate, write_data_yaml  # noqa: E402


def adopt_yolo_dir(src: Path, name: str, origin: str) -> dict:
    """Copy an exported YOLO dataset, keeping its splits (unless --resplit), and record provenance.

    Exported datasets rarely carry sequence ids, so each image is its own sequence and
    leakage between near-duplicate frames cannot be ruled out: reported as a limitation.
    """
    import yaml

    dst = ROOT / "data" / "processed" / name
    if dst.exists():
        shutil.rmtree(dst)
    cand = list(src.rglob("data.yaml"))
    names = ["ball"]
    if cand:
        y = yaml.safe_load(cand[0].read_text())
        names = list(y["names"].values()) if isinstance(y["names"], dict) else list(y["names"])
    elif classes := sorted(src.rglob("classes.txt")):
        names = [n.strip() for n in classes[0].read_text().splitlines() if n.strip()]
    # Normalise ball-class spellings ("cricketBall", "cricket-ball", ...) to "ball".
    names = ["ball" if n.lower().replace("_", "").replace("-", "").replace(" ", "") in {"ball", "cricketball"} else n
             for n in names]
    rows = []
    split_names = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    for img in (p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT and "images" in p.parts):
        parts = list(img.parts)
        k = parts.index("images")
        # Roboflow/Kaggle exports use either <split>/images/x.jpg or images/<split>/x.jpg
        split = next((split_names[q] for q in parts[k - 1:k + 2] if q in split_names), "train")
        lab = Path(*parts[:k], "labels", *parts[k + 1:]).with_suffix(".txt")
        if img.name == "classes.txt":
            continue
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst / "images" / split / img.name)
        if lab.is_file():
            shutil.copy2(lab, dst / "labels" / split / f"{img.stem}.txt")
        rows.append({"file": img.name, "source_sequence": img.stem, "label_origin": origin, "split": split})
    with (dst / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source_sequence", "label_origin", "split"])
        w.writeheader()
        w.writerows(rows)
    write_data_yaml(dst, names)
    return {"images": len(rows), "names": names}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth")
    sub.add_parser("broadcast-seg", help="Acme AI segmentation masks -> YOLO boxes (independent broadcast test set)")
    sub.add_parser("stumps-seg", help="broadcast_seg_test -> single-class stumps set split by match time")
    y = sub.add_parser("yolo-dir")
    y.add_argument("path")
    y.add_argument("--name", required=True)
    y.add_argument("--origin", default="human", help="label origin recorded in provenance.csv")
    y.add_argument("--resplit", action="store_true", help="discard the export's splits; rebuild leak-free splits")
    a = ap.parse_args()
    if a.cmd == "broadcast-seg":
        from ml.datasets.superannotate import NAMES, convert

        src = next((ROOT / "data" / "raw" / "kaggle-semantic-seg").glob("*/images"))
        dst = ROOT / "data" / "processed" / "broadcast_seg_test"
        info = convert(src, dst)
        rep = validate(dst, len(NAMES))
    elif a.cmd == "stumps-seg":
        from ml.datasets.superannotate import derive_stumps_set

        dst = ROOT / "data" / "processed" / "stumps_seg"
        info = derive_stumps_set(ROOT / "data" / "processed" / "broadcast_seg_test", dst)
        rep = validate(dst, 1)
    elif a.cmd == "synth":
        src = ROOT / "data" / "raw" / "hf-cricket-synth"
        dst = ROOT / "data" / "processed" / "synth_yolo"
        info = convert_cricket_synth(src, dst)
        rep = validate(dst, 1)
    else:
        info = adopt_yolo_dir(Path(a.path), a.name, a.origin)
        dst = ROOT / "data" / "processed" / a.name
        if a.resplit:
            tmp = dst.with_name(dst.name + "_orig")
            if tmp.exists():
                shutil.rmtree(tmp)
            dst.rename(tmp)
            info["resplit"] = rebuild_leak_free(tmp, dst)
            shutil.rmtree(tmp)
            write_data_yaml(dst, info["names"])
        rep = validate(dst, len(info["names"]))
    print(json.dumps({"conversion": info, "validation": rep.as_dict()}, indent=2, default=str))
    return 0 if rep.ok else 2


if __name__ == "__main__":
    sys.exit(main())
