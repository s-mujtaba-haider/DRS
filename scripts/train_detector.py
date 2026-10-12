"""Fine-tune a YOLO ball detector (Ultralytics) with pre-flight checks and run records.

    # smoke test (a couple of epochs, small images) - verifies the pipeline, NOT a usable model
    python scripts/train_detector.py --data data/processed/synth_yolo/data.yaml --smoke
    # baseline
    python scripts/train_detector.py --data data/processed/ball_yolo/data.yaml --epochs 80 --imgsz 1280 --batch 8
    # resume an interrupted run
    python scripts/train_detector.py --resume models/runs/<name>/weights/last.pt

Checks dataset paths, annotation validity and class mapping, GPU availability, and records
config + library versions next to the run. The best checkpoint is copied to
models/checkpoints/cricket_ball_best.pt only with --promote (so an untested model is never used
automatically), together with a JSON model card.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="YOLO data.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained weights to fine-tune (COCO, AGPL-3.0)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8, help="reduce if CUDA OOM; -1 lets Ultralytics pick")
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers (keep low with 16 GB RAM)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true", help="2 epochs at 640px: pipeline check only")
    ap.add_argument("--resume", help="path to last.pt of an interrupted run")
    ap.add_argument("--promote", action="store_true", help="copy best.pt to models/checkpoints/cricket_ball_best.pt")
    ap.add_argument("--purpose", default="", help="free-text note stored in the model card")
    a = ap.parse_args()

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO

    from ml.datasets.yolo_tools import validate

    runs = ROOT / "models" / "runs"
    if a.resume:
        model = YOLO(a.resume)
        model.train(resume=True)
        return 0
    if not a.data:
        ap.error("--data is required unless --resume is given")

    data_yaml = Path(a.data)
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    if "ball" not in [n.lower() for n in names]:
        print(f"ERROR: class list {names} has no 'ball' class")
        return 2
    root = Path(cfg.get("path") or data_yaml.parent)
    rep = validate(root, len(names), check_duplicates=False)
    print(json.dumps({k: v for k, v in rep.as_dict().items() if k in ("images", "boxes", "class_counts", "per_split", "box_size_px", "ok",
                                                                      "num_invalid_lines", "split_leaks")}, indent=1))
    if not rep.ok:
        print("ERROR: dataset validation failed (invalid lines, unknown classes or split leakage)")
        return 2
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available - training on CPU will be very slow")
        a.device = "cpu"
    else:
        free, total = torch.cuda.mem_get_info()
        print(f"GPU {torch.cuda.get_device_name(0)}: {free / 1e9:.1f}/{total / 1e9:.1f} GB free")

    if a.smoke:
        a.epochs, a.imgsz, a.batch = 2, 640, min(a.batch, 4) if a.batch > 0 else 4
    name = a.name or f"{'smoke' if a.smoke else 'ball'}_{time.strftime('%Y%m%d_%H%M%S')}"
    record = {"args": vars(a), "dataset_validation": rep.as_dict(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "ultralytics": ultralytics.__version__, "python": platform.python_version(), "platform": platform.platform(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (runs / name).mkdir(parents=True, exist_ok=True)
    (runs / name / "run_record.json").write_text(json.dumps(record, indent=2, default=str))

    model = YOLO(str(ROOT / "models" / "checkpoints" / a.model) if (ROOT / "models" / "checkpoints" / a.model).is_file() else a.model)
    t0 = time.time()
    model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
                project=str(runs), name=name, exist_ok=True, amp=True, cache=False, patience=20, plots=True,
                # small-object friendly augmentation: no heavy downscaling, keep mosaic early only
                scale=0.3, mosaic=1.0, close_mosaic=10, seed=0, deterministic=True)
    best = runs / name / "weights" / "best.pt"
    metrics = model.val(data=str(data_yaml), split="val", imgsz=a.imgsz, device=a.device, plots=False,
                        project=str(runs), name=f"{name}_val", exist_ok=True)
    card = {
        "name": name, "base_model": a.model, "classes": names, "imgsz": a.imgsz, "epochs": a.epochs,
        "train_time_s": round(time.time() - t0, 1), "smoke_test": a.smoke, "purpose": a.purpose,
        "val_metrics": {k: float(v) for k, v in metrics.results_dict.items()},
        "dataset": str(data_yaml), "dataset_label_origins": "see provenance.csv",
        "warning": "Validation metrics only; evaluate on a held-out test split (scripts/evaluate_detector.py) "
                   "and on real target footage before use.",
    }
    (runs / name / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    if a.promote:
        if a.smoke:
            print("Refusing to promote a smoke-test model")
            return 3
        dst = ROOT / "models" / "checkpoints" / "cricket_ball_best.pt"
        shutil.copy2(best, dst)
        shutil.copy2(runs / name / "model_card.json", dst.with_suffix(".json"))
        print(f"Promoted to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
