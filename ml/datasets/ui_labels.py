"""Export human labels made in the web console (Annotation rows) as a YOLO labelled set.

The output layout matches ``scripts/annotate_video.py export`` (images/, labels/, classes.txt,
provenance.csv) so ``annotate_video.py build`` can merge it with other sets and split by video.
Only the latest annotation per frame counts. ``mark_absent`` frames become negatives (empty label
files); ``reject`` rows and frames without a human decision are not exported.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from ml.datasets.yolo_tools import DRS_CLASSES
from ml.video.ingest import iter_frames

PROV_FIELDS = ["file", "source_sequence", "label_origin", "frame_index", "timestamp_ms"]


def latest_labels(annotations: list[dict], label: str = "ball") -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for a in sorted(annotations, key=lambda r: r["created_at"]):
        if a.get("label", "ball") == label:
            latest[a["frame_index"]] = a
    return {f: a for f, a in latest.items() if a["action"] in ("add", "move", "mark_absent")}


def export_ui_labels(video_path: str | Path, sequence: str, width: int, height: int, annotations: list[dict],
                     out: str | Path, default_box_px: float = 12.0) -> dict:
    """Write images/labels for every human-labelled frame of one video. Returns counts."""
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(DRS_CLASSES) + "\n")
    labels = latest_labels(annotations)
    prov_path = out / "provenance.csv"
    rows = list(csv.DictReader(prov_path.open())) if prov_path.is_file() else []
    rows = [r for r in rows if r["source_sequence"] != sequence]
    pos = neg = 0
    wanted = sorted(labels)
    if wanted:
        frames = {rec.index: rec for rec in iter_frames(video_path, start_frame=wanted[0], end_frame=wanted[-1] + 1)
                  if rec.index in labels}
    else:
        frames = {}
    for fidx in wanted:
        rec = frames.get(fidx)
        if rec is None:
            continue
        a = labels[fidx]
        name = f"{sequence}_f{fidx:06d}"
        cv2.imwrite(str(out / "images" / f"{name}.png"), rec.image)
        lines = []
        if a["action"] in ("add", "move"):
            d = a["data"]
            r = d.get("radius_px") or default_box_px / 2
            lines.append(f"0 {d['x'] / width:.6f} {d['y'] / height:.6f} {2 * r / width:.6f} {2 * r / height:.6f}")
            pos += 1
        else:
            neg += 1
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
        rows.append({"file": f"{name}.png", "source_sequence": sequence, "label_origin": "human", "frame_index": fidx,
                     "timestamp_ms": f"{rec.timestamp_ms:.3f}"})
    with prov_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PROV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return {"sequence": sequence, "positives": pos, "negatives": neg, "frames": pos + neg}
