"""Export the cricket-ball detector to a TensorRT FP16 engine and validate it against the .pt model.

    python scripts/export_tensorrt.py [--weights models/checkpoints/cricket_ball_best.pt] [--imgsz 1280]

The engine is written next to the weights (``<stem>.engine``). It is then run side by side with the
PyTorch model on the independent broadcast test frames. Detections are matched (IoU >= 0.5), and
both models are scored against the ground-truth labels. A sidecar ``<stem>.engine.json`` records the
source hash, TensorRT version, GPU, build settings and the agreement metrics. The detector only uses
the engine when that sidecar says validation passed and still matches the installed environment
(``ml.detection.yolo_detector.validated_engine``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Never let Ultralytics pip-install packages into the venv mid-export (it once replaced numpy).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# Pass criteria (engine vs PyTorch FP16, same images, same confidence threshold).
MIN_AGREEMENT = 0.97  # share of PyTorch detections (score >= 0.25) the engine reproduces
MAX_EXTRA = 0.03  # engine-only detections (score >= 0.25), relative to PyTorch detections
MAX_MEDIAN_CENTRE_PX = 1.0
MAX_AP_DROP = 0.01  # AP against labels may not drop by more than this


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def compare(ref: list[list[tuple]], eng: list[list[tuple]], thr: float = 0.25) -> dict:
    """Greedy IoU matching of engine boxes to reference boxes per image (both: (x1,y1,x2,y2,score))."""
    n_ref = n_eng = matched = 0
    centre, ious, dscore = [], [], []
    for R, E in zip(ref, eng):
        R = [r for r in R if r[4] >= thr]
        E = [e for e in E if e[4] >= thr]
        n_ref, n_eng = n_ref + len(R), n_eng + len(E)
        used = set()
        for r in sorted(R, key=lambda r: -r[4]):
            best, bi = 0.5, None
            for j, e in enumerate(E):
                if j not in used and (v := _iou(r, e)) >= best:
                    best, bi = v, j
            if bi is None:
                continue
            used.add(bi)
            matched += 1
            e = E[bi]
            centre.append(float(np.hypot((r[0] + r[2] - e[0] - e[2]) / 2, (r[1] + r[3] - e[1] - e[3]) / 2)))
            ious.append(best)
            dscore.append(abs(r[4] - e[4]))
    return {"score_threshold": thr, "reference_detections": n_ref, "engine_detections": n_eng, "matched": matched,
            "agreement": matched / n_ref if n_ref else 1.0,
            "engine_extra_rate": (n_eng - matched) / n_ref if n_ref else 0.0,
            "centre_err_px_median": float(np.median(centre)) if centre else None,
            "centre_err_px_p95": float(np.percentile(centre, 95)) if centre else None,
            "iou_median": float(np.median(ious)) if ious else None,
            "score_abs_diff_median": float(np.median(dscore)) if dscore else None,
            "score_abs_diff_max": float(np.max(dscore)) if dscore else None}


def run_on_images(det, imgs, labels_dir: Path, thr: float):
    import cv2

    from ml.detection.base import FrameInfo
    from ml.evaluation.detection_metrics import match_frame, summarize

    preds_all, per_frame = [], []
    for i, p in enumerate(imgs):
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        gts = []
        lab = labels_dir / f"{p.stem}.txt"
        for line in lab.read_text().splitlines() if lab.is_file() else []:
            c, cx, cy, w, h = line.split()[:5]
            if int(c) == 0:
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                gts.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        preds = [(*d.bbox, d.confidence) for d in det.detect_batch([im], [FrameInfo(i, 0.0, "container_pts")])[0]]
        preds_all.append(preds)
        per_frame.append(match_frame(preds, gts))
    s = summarize(per_frame, thr)
    keys = ("num_gt", "precision", "recall", "f1", "ap_centre_match", "true_positives", "false_positives", "false_negatives")
    return preds_all, {k: s[k] for k in keys if k in s}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/checkpoints/cricket_ball_best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280, help="production inference size (PipelineConfig.imgsz)")
    ap.add_argument("--max-hw", default="736,1280",
                    help="largest letterboxed input (h,w) in the dynamic profile. 736x1280 covers 16:9 frames at "
                         "imgsz 1280 and needs about half the activation memory of a square 1280x1280 profile. "
                         "Frames that letterbox larger are sent to the .pt model by the detector.")
    ap.add_argument("--max-batch", type=int, default=8, help="PipelineConfig.batch_size; activation memory scales with it")
    ap.add_argument("--workspace-gb", type=float, default=4.0)
    ap.add_argument("--data", default=str(ROOT / "data/processed/broadcast_seg_test"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.01)
    a = ap.parse_args()

    import tensorrt
    import torch
    from ultralytics import YOLO

    from ml.detection.yolo_detector import YoloBallDetector, file_sha256

    if not torch.cuda.is_available():
        print("CUDA is required to build a TensorRT engine")
        return 1
    w = Path(a.weights)
    t0 = time.perf_counter()
    max_hw = [int(v) for v in a.max_hw.split(",")]
    if len(max_hw) != 2 or max(max_hw) != a.imgsz or any(v % 32 for v in max_hw):
        print("--max-hw must be two multiples of 32 whose larger value equals --imgsz")
        return 1
    built = Path(YOLO(str(w)).export(format="engine", half=True, dynamic=True, batch=a.max_batch, imgsz=max_hw,
                                     workspace=a.workspace_gb, device=0, verbose=False))
    build_s = time.perf_counter() - t0
    engine = w.with_suffix(".engine")
    if built.resolve() != engine.resolve():
        shutil.move(str(built), engine)
    sidecar = engine.with_suffix(".engine.json")
    sidecar.unlink(missing_ok=True)  # never leave a stale "passed" record for a new engine

    root = Path(a.data)
    imgs = sorted(p for p in (root / "images" / a.split).glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    labels = root / "labels" / a.split
    ref_det = YoloBallDetector(w, conf=a.conf, imgsz=a.imgsz, batch_size=1, half=True)
    eng_det = YoloBallDetector(engine, conf=a.conf, imgsz=a.imgsz, batch_size=1, half=True)
    ref_preds, ref_m = run_on_images(ref_det, imgs, labels, 0.25)
    eng_preds, eng_m = run_on_images(eng_det, imgs, labels, 0.25)
    agree = compare(ref_preds, eng_preds)
    if "ap_centre_match" not in ref_m or "ap_centre_match" not in eng_m:
        print("No labelled balls in the validation split: cannot check AP")
        return 1
    ap_drop = ref_m["ap_centre_match"] - eng_m["ap_centre_match"]
    checks = {
        "agreement": agree["agreement"] >= MIN_AGREEMENT,
        "extra": agree["engine_extra_rate"] <= MAX_EXTRA,
        "centre": (agree["centre_err_px_median"] or 0) <= MAX_MEDIAN_CENTRE_PX,
        "ap": ap_drop <= MAX_AP_DROP,
    }
    meta = {
        "schema": "drs.trt_engine/1", "created_at": datetime.now(UTC).isoformat(),
        "source_weights": w.name, "source_sha256": file_sha256(w), "engine": engine.name,
        "engine_sha256": file_sha256(engine), "tensorrt": tensorrt.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "precision": "fp16", "dynamic": True, "imgsz": a.imgsz, "max_hw": max_hw,
        "max_batch": a.max_batch, "build_seconds": round(build_s, 1),
        "validation": {"data": str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
                       "split": a.split, "images": len(imgs), "pytorch_vs_labels": ref_m, "engine_vs_labels": eng_m,
                       "agreement": agree, "ap_drop": ap_drop,
                       "criteria": {"min_agreement": MIN_AGREEMENT, "max_extra_rate": MAX_EXTRA,
                                    "max_median_centre_px": MAX_MEDIAN_CENTRE_PX, "max_ap_drop": MAX_AP_DROP},
                       "checks": checks, "passed": all(checks.values())},
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    out = ROOT / "outputs/reports" / f"trt_export_{w.stem}_{a.imgsz}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["validation"], indent=2))
    print(f"engine {engine} ({'VALIDATED' if meta['validation']['passed'] else 'NOT validated: the detector will keep using the .pt'})")
    return 0 if meta["validation"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
