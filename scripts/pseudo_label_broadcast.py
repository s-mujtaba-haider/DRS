"""Track-verified pseudo-labels from unlabelled broadcast clips (self-training).

A detector + tracker are run over every allowed clip; only detections that belong to tracks
passing the strict consistency checks (enough observations, smooth Kalman innovations,
minimum confidence) become labels. A ball-like object that moves consistently for many
frames is far more likely to be the ball than a single detection.

Labels are recorded as ``label_origin = auto_track_verified``. They are NOT ground truth:
review samples (``--preview``) and always evaluate on the independent human-labelled
broadcast test set, which this script never touches (series with --exclude-team are skipped).

    python scripts/pseudo_label_broadcast.py --name ball_pseudo_v1 --preview 48
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets.composite import summarize_rows, write_provenance, write_yolo_sample  # noqa: E402
from ml.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from ml.video.ingest import iter_frames  # noqa: E402
from scripts.build_broadcast_training_set import series_of_clips  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ball_pseudo_v1")
    ap.add_argument("--weights", default=str(ROOT / "models" / "checkpoints" / "cricket_ball_best.pt"))
    ap.add_argument("--val-series", default="New Zealand v Pakistan")
    ap.add_argument("--exclude-team", default="Zimbabwe")
    ap.add_argument("--min-observed", type=int, default=8)
    ap.add_argument("--max-innovation", type=float, default=9.0, help="stricter than the app's 12 px reliability check")
    ap.add_argument("--min-mean-conf", type=float, default=0.25)
    ap.add_argument("--crop", type=int, default=960)
    ap.add_argument("--every", type=int, default=2, help="use every Nth verified frame (neighbours are near-duplicates)")
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    clips = ROOT / "data" / "raw" / "hf-cricket-delivery-videos"
    # Cache per model so a new round never reuses detections from older weights.
    work = ROOT / "outputs" / "pseudo_tracking" / Path(a.weights).stem
    dst = ROOT / "data" / "processed" / a.name
    if dst.exists():
        shutil.rmtree(dst)
    series = series_of_clips(clips)
    cfg = PipelineConfig(detector="cricket", weights=a.weights, conf=0.15, imgsz=1280, batch_size=16, render_video=False,
                         max_frames=None)
    rows: list[dict] = []
    previews: list[np.ndarray] = []
    stats = {"clips": 0, "verified_tracks": 0, "labels": 0}
    for v in sorted((clips / "videos").glob("*.mp4")):
        s = series.get(v.stem)
        if s is None or a.exclude_team in s:
            continue
        split = "val" if s == a.val_series else "train"
        out = work / v.stem
        if not (out / "tracking.json").is_file():
            run_pipeline(v, out, cfg)
        stats["clips"] += 1
        tracking = json.loads((out / "tracking.json").read_text())
        dets = {}
        for fd in json.loads((out / "detections.json").read_text()):
            for d in fd["detections"]:
                dets[(d["frame_index"], round(d["center"][0], 2), round(d["center"][1], 2))] = d
        labels: dict[int, list[list[float]]] = {}
        for tr in tracking["tracks"]:
            q = tr["quality"]
            if (q["num_observed"] < a.min_observed or (q["mean_innovation_px"] or 1e9) > a.max_innovation
                    or (q["mean_confidence"] or 0) < a.min_mean_conf):
                continue
            stats["verified_tracks"] += 1
            for p in tr["points"]:
                if p["kind"] != "observed":
                    continue
                d = dets.get((p["frame_index"], round(p["x"], 2), round(p["y"], 2)))
                if d:
                    labels.setdefault(p["frame_index"], []).append(d["bbox"])
        keep = sorted(labels)[:: a.every]
        if not keep:
            print(f"\r{v.stem} {s:<28} labels={stats['labels']}", end="", flush=True)
            continue
        want = set(keep)
        for rec in iter_frames(v):
            if rec.index not in want:
                continue
            H, W = rec.image.shape[:2]
            boxes = labels[rec.index]
            bx = (boxes[0][0] + boxes[0][2]) / 2
            by = (boxes[0][1] + boxes[0][3]) / 2
            cw, ch = min(a.crop, W), min(a.crop, H)
            x0 = int(np.clip(bx - rng.uniform(0.15, 0.85) * cw, 0, W - cw))
            y0 = int(np.clip(by - rng.uniform(0.15, 0.85) * ch, 0, H - ch))
            crop = rec.image[y0:y0 + ch, x0:x0 + cw]
            inside = [(b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0) for b in boxes
                      if b[0] >= x0 and b[1] >= y0 and b[2] <= x0 + cw and b[3] <= y0 + ch]
            if not inside:
                continue
            write_yolo_sample(dst, split, f"pseudo_{v.stem}_f{rec.index:05d}", crop, inside, rows, sequence=f"{s}|{v.stem}")
            rows[-1]["label_origin"] = "auto_track_verified"
            stats["labels"] += len(inside)
            if a.preview and len(previews) < a.preview and rng.random() < 0.3:
                x1, y1, x2, y2 = map(int, inside[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                patch = crop[max(0, cy - 48):cy + 48, max(0, cx - 48):cx + 48].copy()
                if patch.shape[0] == 96 and patch.shape[1] == 96:
                    previews.append(cv2.resize(patch, (144, 144), interpolation=cv2.INTER_NEAREST))
        print(f"\r{v.stem} {s:<28} tracks={stats['verified_tracks']} labels={stats['labels']}", end="", flush=True)
    print()
    write_provenance(dst, rows)
    if previews:
        n = 8
        while len(previews) % n:
            previews.append(np.zeros_like(previews[0]))
        grid = np.vstack([np.hstack(previews[i:i + n]) for i in range(0, len(previews), n)])
        cv2.imwrite(str(dst / "preview_grid.jpg"), grid)
    print(json.dumps({**stats, **summarize_rows(rows)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
