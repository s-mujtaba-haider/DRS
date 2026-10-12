"""Evaluate a ball detector on a YOLO-format split with centre-distance matching.

    python scripts/evaluate_detector.py --data data/processed/synth_yolo/data.yaml --split test \
        --weights models/runs/<run>/weights/best.pt --name synth_smoke

Reports precision, recall, F1, AP (centre match), localisation error and small-object
statistics; writes outputs/reports/detector_eval_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.25, help="score threshold for P/R/F1")
    ap.add_argument("--match-px", type=float, default=6.0)
    ap.add_argument("--tile", type=int, default=None, help="tiled native-resolution inference (tile size px)")
    ap.add_argument("--name", default="eval")
    a = ap.parse_args()

    import cv2
    import yaml

    from ml.detection.base import FrameInfo
    from ml.detection.yolo_detector import YoloBallDetector
    from ml.evaluation.detection_metrics import match_frame, summarize

    cfg = yaml.safe_load(Path(a.data).read_text())
    root = Path(cfg.get("path") or Path(a.data).parent)
    img_dir = root / "images" / a.split
    imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    if not imgs:
        print(f"No images in {img_dir}")
        return 1
    det = YoloBallDetector(a.weights, conf=a.conf, imgsz=a.imgsz, batch_size=8 if a.tile else 1, tile_size=a.tile)
    per_frame, sizes = [], []
    for i, p in enumerate(imgs):
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        gts = []
        lab = root / "labels" / a.split / f"{p.stem}.txt"
        for line in lab.read_text().splitlines() if lab.is_file() else []:
            c, cx, cy, w, h = line.split()[:5]
            if int(c) != 0:
                continue
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            gts.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            sizes.append((w * h) ** 0.5)
        preds = [(*d.bbox, d.confidence) for d in det.detect_batch([im], [FrameInfo(i, 0.0, "container_pts")])[0]]
        per_frame.append(match_frame(preds, gts, a.match_px))
    summary = summarize(per_frame, a.threshold, sizes)
    summary.update(weights=a.weights, data=a.data, split=a.split, images=len(imgs), model_id=det.model_id,
                   imgsz=a.imgsz, tile=a.tile,
                   caveat="Metrics apply only to this split's distribution; they do not establish accuracy on other footage.")
    out = ROOT / "outputs" / "reports" / f"detector_eval_{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
