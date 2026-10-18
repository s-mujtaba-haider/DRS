"""Qualitative evaluation of batter ground-contact proposals on real broadcast delivery clips.

There is no labelled run-out / stumping footage, so this measures what can be measured without
ground truth: how often a batter is selected near the striker's wicket, keypoint confidence,
grounded-hint distribution and inference speed. It also writes overlay images for visual checks.

The striker's wicket is found with the stumps detector (the far, smaller box in a
behind-the-bowler view) and a region around it is passed to the proposer, as an operator
would draw it.

    python scripts/evaluate_crease_proposals.py --every 10 --frames 75
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.runout.proposals import UltralyticsPose, propose_in_video  # noqa: E402
from ml.video.ingest import iter_frames  # noqa: E402


def striker_roi(stumps_model, img: np.ndarray) -> tuple[list[float] | None, list[float] | None]:
    r = stumps_model.predict(img, imgsz=1280, conf=0.3, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None
    b = r.boxes.xyxy.cpu().numpy()
    far = b[np.argmin(b[:, 3])]  # higher in the image = farther = striker's end
    w, h = far[2] - far[0], far[3] - far[1]
    H, W = img.shape[:2]
    roi = [max(0, far[0] - 4 * w), max(0, far[1] - 1.5 * h), min(W, far[2] + 4 * w), min(H, far[3] + 1.2 * h)]
    return [float(v) for v in far], [float(v) for v in roi]


def draw(img, fr, roi, stumps):
    out = img.copy()
    if roi:
        cv2.rectangle(out, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (255, 0, 255), 2)
    if stumps:
        cv2.rectangle(out, (int(stumps[0]), int(stumps[1])), (int(stumps[2]), int(stumps[3])), (255, 255, 0), 1)
    if fr["person_box"]:
        b = fr["person_box"]
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)
    col = {True: (0, 255, 0), False: (0, 0, 255), None: (0, 165, 255)}
    for p in fr["proposals"]:
        x, y = map(int, p["point"])
        cv2.circle(out, (x, y), 4 if p["kind"] != "toe_est" else 2, col[p["grounded_hint"]], -1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "data/raw/hf-cricket-delivery-videos/videos"))
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--frames", type=int, default=75)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default=str(ROOT / "outputs/reports/crease_proposals_eval.json"))
    ap.add_argument("--viz", default=str(ROOT / "outputs/reports/crease_proposals_viz"))
    args = ap.parse_args()

    from ultralytics import YOLO

    stumps = YOLO(str(ROOT / "models/checkpoints/cricket_stumps_best.pt"))
    pose = UltralyticsPose(imgsz=args.imgsz, batch_size=16)
    vids = sorted(Path(args.videos).glob("*.mp4"))[:: args.every]
    viz = Path(args.viz)
    viz.mkdir(parents=True, exist_ok=True)
    clips, confs, hints, fps = [], [], {"true": 0, "false": 0, "none": 0}, []
    t_all = time.perf_counter()
    for v in vids:
        st = roi = None
        for rec in iter_frames(v, max_frames=10, step=2):
            st, roi = striker_roi(stumps, rec.image)
            if roi:
                break
        if roi is None:
            clips.append({"clip": v.name, "stumps_found": False})
            print(f"{v.name}: no striker stumps found in the first 10 frames, skipped")
            continue
        res = propose_in_video(v, pose, 0, args.frames, roi=tuple(roi), batch=16)
        d = res.model_dump(mode="json")
        if res.inference_fps:
            fps.append(res.inference_fps)
        for f in d["frames"]:
            for p in f["proposals"]:
                if p["kind"].startswith("ankle"):
                    confs.append(p["confidence"])
                    hints["none" if p["grounded_hint"] is None else str(p["grounded_hint"]).lower()] += 1
        mid = args.frames // 3
        img = next(iter_frames(v, start_frame=mid, max_frames=1)).image
        fr = next((f for f in d["frames"] if f["frame_index"] == mid), None)
        if fr:
            cv2.imwrite(str(viz / f"{v.stem}_f{mid}.jpg"), draw(img, fr, roi, st), [cv2.IMWRITE_JPEG_QUALITY, 85])
        clips.append({"clip": v.name, "stumps_found": st is not None, "roi": roi,
                      "frames": len(d["frames"]), "frames_with_batter": d["frames_with_batter"],
                      "mean_people_per_frame": float(np.mean([f["n_people"] for f in d["frames"]])) if d["frames"] else 0,
                      "inference_fps": res.inference_fps})
        print(f"{v.name}: stumps={'y' if st else 'n'} batter {d['frames_with_batter']}/{len(d['frames'])} fps={res.inference_fps}")

    # CPU throughput on a few frames
    cpu = UltralyticsPose(imgsz=args.imgsz, device="cpu", batch_size=4)
    imgs = [r.image for r in iter_frames(vids[0], max_frames=9)][1:]
    cpu.predict(imgs[:1])  # warm-up
    t0 = time.perf_counter()
    cpu.predict(imgs)
    cpu_fps = len(imgs) / (time.perf_counter() - t0)

    c = np.array(confs) if confs else np.array([np.nan])
    with_roi = [x for x in clips if x["stumps_found"]]
    summary = {
        "date": time.strftime("%Y-%m-%d"), "model": pose.model_id, "licence": "AGPL-3.0",
        "data": f"hf-cricket-delivery-videos, every {args.every}th clip, first {args.frames} frames, 1920x1080 25 fps broadcast",
        "ground_truth": "none: no labelled run-out/stumping footage; numbers below are not accuracy",
        "clips": len(clips), "clips_with_striker_stumps": len(with_roi),
        "selection": "person nearest a region around the striker's stumps (stumps detector); clips without stumps skipped",
        "no_region_fallback_finding": ("An earlier run without a region selected the largest person: on 16 clips without "
                                       "stumps it picked the umpire, non-striker, bowler or a TV-graphic face in at least 7 "
                                       "(visual check), so the fallback was removed and a crease or region is now required"),
        "batter_found_frame_rate_with_roi": (sum(x["frames_with_batter"] for x in with_roi) / max(1, sum(x["frames"] for x in with_roi))),
        "ankle_confidence": {"n": int(len(confs)), "p10": float(np.nanpercentile(c, 10)), "median": float(np.nanmedian(c)),
                             "p90": float(np.nanpercentile(c, 90)), "frac_ge_0.5": float(np.mean(c >= 0.5)) if confs else None},
        "grounded_hints": hints,
        "gpu_inference_fps_median": float(np.median(fps)) if fps else None,
        "cpu_inference_fps": round(cpu_fps, 2),
        "wall_s": round(time.perf_counter() - t_all, 1),
        "per_clip": clips,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_clip"}, indent=2))


if __name__ == "__main__":
    main()
