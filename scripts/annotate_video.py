"""Annotation workflow helpers for building a cricket dataset from your own footage.

1) Export frames for labelling (YOLO layout, optionally pre-labelled by a model):
    python scripts/annotate_video.py export VIDEO --out data/annotations/clip1 --step 2 [--prelabel WEIGHTS]
   Label with any YOLO-compatible tool (e.g. CVAT, Label Studio, makesense.ai) using
   classes from data/annotations/<set>/classes.txt. Pre-labels are marked label_origin=auto
   in provenance.csv; change to auto_corrected / human after review.

2) Export manual corrections made in the web UI (stored in the backend DB) as YOLO labels:
    python scripts/annotate_video.py from-ui --video-id ID --out data/annotations/ui_ball --api http://127.0.0.1:8100

3) Merge labelled sets into a training dataset split by source video:
    python scripts/annotate_video.py build --sets data/annotations/clip1 data/annotations/clip2 --name ball_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets.yolo_tools import DRS_CLASSES, split_by_sequence, validate, write_data_yaml  # noqa: E402
from ml.video.ingest import iter_frames  # noqa: E402

PROV_FIELDS = ["file", "source_sequence", "label_origin", "frame_index", "timestamp_ms"]


def cmd_export(a) -> int:
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(DRS_CLASSES) + "\n")
    det = None
    if a.prelabel:
        from ml.detection.yolo_detector import YoloBallDetector

        det = YoloBallDetector(a.prelabel, conf=0.25)
    seq = Path(a.video).stem
    rows = []
    for rec in iter_frames(a.video, start_frame=a.start, end_frame=a.end, step=a.step):
        name = f"{seq}_f{rec.index:06d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), rec.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        origin = "unlabeled"
        if det is not None:
            from ml.detection.base import FrameInfo

            dets = det.detect_batch([rec.image], [FrameInfo(rec.index, rec.timestamp_ms, rec.timestamp_source)])[0]
            H, W = rec.image.shape[:2]
            lines = [f"0 {d.center[0] / W:.6f} {d.center[1] / H:.6f} {(d.bbox[2] - d.bbox[0]) / W:.6f} {(d.bbox[3] - d.bbox[1]) / H:.6f}"
                     for d in dets]
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
            origin = "auto"
        rows.append({"file": f"{name}.jpg", "source_sequence": seq, "label_origin": origin, "frame_index": rec.index,
                     "timestamp_ms": f"{rec.timestamp_ms:.3f}"})
    with (out / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PROV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Exported {len(rows)} frames to {out}")
    return 0


def cmd_from_ui(a) -> int:
    base = a.api.rstrip("/")
    video = json.load(urllib.request.urlopen(f"{base}/api/videos/{a.video_id}"))
    anns = json.load(urllib.request.urlopen(f"{base}/api/videos/{a.video_id}/annotations"))
    latest: dict[int, dict] = {}
    for x in sorted(anns, key=lambda r: r["created_at"]):
        latest[x["frame_index"]] = x
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    W, H = video["meta"]["width"], video["meta"]["height"]
    seq = Path(video["original_filename"]).stem
    rows = []
    r = a.box_px / 2
    for fidx, x in sorted(latest.items()):
        if x["action"] not in ("add", "move", "mark_absent"):
            continue
        with urllib.request.urlopen(f"{base}/api/videos/{a.video_id}/frame/{fidx}") as resp:
            (out / "images" / f"{seq}_f{fidx:06d}.png").write_bytes(resp.read())
        lines = []
        if x["action"] in ("add", "move"):
            cx, cy = x["data"]["x"], x["data"]["y"]
            rr = x["data"].get("radius_px") or r
            lines.append(f"0 {cx / W:.6f} {cy / H:.6f} {2 * rr / W:.6f} {2 * rr / H:.6f}")
        (out / "labels" / f"{seq}_f{fidx:06d}.txt").write_text("\n".join(lines))
        rows.append({"file": f"{seq}_f{fidx:06d}.png", "source_sequence": seq, "label_origin": "human", "frame_index": fidx,
                     "timestamp_ms": ""})
    with (out / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PROV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Exported {len(rows)} human-labelled frames (ball boxes use a {a.box_px}px default size where no radius was given)")
    return 0


def cmd_build(a) -> int:
    dst = ROOT / "data" / "processed" / a.name
    if dst.exists():
        shutil.rmtree(dst)
    items: dict[str, list[tuple[Path, Path, dict]]] = {}
    for s in map(Path, a.sets):
        prov = {r["file"]: r for r in csv.DictReader((s / "provenance.csv").open())}
        for img in sorted((s / "images").glob("*")):
            p = prov.get(img.name, {"source_sequence": s.name, "label_origin": "unknown"})
            if p.get("label_origin") == "unlabeled" and not a.include_unlabeled:
                continue
            if p.get("label_origin") == "auto" and not a.include_auto:
                continue  # unreviewed model output is not ground truth
            items.setdefault(p["source_sequence"], []).append((img, s / "labels" / f"{img.stem}.txt", p))
    assign = split_by_sequence({k: [str(i[0]) for i in v] for k, v in items.items()}, seed=a.seed)
    rows = []
    for seqn, lst in items.items():
        for img, lab, p in lst:
            split = assign[str(img)]
            (dst / "images" / split).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst / "images" / split / img.name)
            if lab.is_file():
                shutil.copy2(lab, dst / "labels" / split / lab.name)
            rows.append({"file": img.name, "source_sequence": seqn, "label_origin": p.get("label_origin"), "split": split})
    with (dst / "provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source_sequence", "label_origin", "split"])
        w.writeheader()
        w.writerows(rows)
    write_data_yaml(dst, DRS_CLASSES)
    rep = validate(dst, len(DRS_CLASSES))
    print(json.dumps(rep.as_dict(), indent=1, default=str)[:3000])
    return 0 if rep.ok else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("video")
    e.add_argument("--out", required=True)
    e.add_argument("--step", type=int, default=2)
    e.add_argument("--start", type=int, default=0)
    e.add_argument("--end", type=int, default=None)
    e.add_argument("--prelabel")
    u = sub.add_parser("from-ui")
    u.add_argument("--video-id", required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--api", default="http://127.0.0.1:8100")
    u.add_argument("--box-px", type=float, default=10.0)
    b = sub.add_parser("build")
    b.add_argument("--sets", nargs="+", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--include-auto", action="store_true", help="include unreviewed pre-labels (not recommended)")
    b.add_argument("--include-unlabeled", action="store_true", help="include unlabeled frames as negatives")
    a = ap.parse_args()
    return {"export": cmd_export, "from-ui": cmd_from_ui, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
