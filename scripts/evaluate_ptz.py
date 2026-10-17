"""Check broadcast pan/tilt/zoom compensation on real clips with an independent static landmark.

For each clip, the delivery window (first frames, up to the first cut) is stabilised to its middle frame
(ml/video/ptz.py). The wickets do not move, so their bases, detected independently in every frame
by the stumps detector, should land on the same reference pixel after mapping. The residual of
each mapped base from its landmark (and of the raw, unmapped base for comparison) measures the
compensation error, including stumps-detector noise (about 1-2 px).

    python scripts/evaluate_ptz.py --clips data/raw/hf-cricket-delivery-videos/videos --n 20 \
        --out outputs/reports/ptz_stabilisation_real.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.video.ptz import estimate_shot_motion, read_segment  # noqa: E402


def delivery_window(video: str, length: int) -> tuple[int, int]:
    """The first ``length`` frames, cut short at the first shot cut. In these clips the delivery
    (run-up, release, bounce, impact) happens in the first 1-2 s of the delivery shot; later frames
    often follow the ball after the shot with fast pans in which the far wicket leaves the frame."""
    from ml.video.shots import detect_shot_cuts

    cuts = detect_shot_cuts(video, max_frames=length)
    return 0, (cuts[0] - 1) if cuts else length - 1


def landmark_residuals(P: np.ndarray, radius: float = 40.0, min_members: int = 5) -> np.ndarray:
    """Distance of each stumps base to the centre of its static landmark.

    Bases are greedily grouped (within ``radius`` px of a group's running median); groups with
    fewer than ``min_members`` detections are false detections and are dropped. Both wickets
    (and any detector duplicates) become separate landmarks, so no wicket labelling is needed.
    """
    groups: list[list[np.ndarray]] = []
    for p in P:
        for g in groups:
            if np.linalg.norm(np.median(g, 0) - p) <= radius:
                g.append(p)
                break
        else:
            groups.append([p])
    res = [np.linalg.norm(np.asarray(g) - np.median(g, 0), axis=1) for g in groups if len(g) >= min_members]
    return np.concatenate(res) if res else np.array([np.nan])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--window", type=int, default=45, help="frames from the clip start (delivery window)")
    ap.add_argument("--weights", default=str(ROOT / "models" / "checkpoints" / "cricket_stumps_best.pt"))
    ap.add_argument("--out")
    a = ap.parse_args()
    from ultralytics import YOLO

    from ml.calibration.auto import detect_stumps

    model = YOLO(a.weights)
    rows = []
    for video in sorted(Path(a.clips).glob("*.mp4"))[:: a.every][: a.n]:
        s, e = delivery_window(str(video), a.window)
        frames = read_segment(str(video), s, e)
        if len(frames) < 10:
            continue
        ref = sorted(frames)[len(frames) // 2]
        motion = estimate_shot_motion(frames, ref)
        raw, mapped, zooms = [], [], []
        for i, f in frames.items():
            m_ok = i in motion.to_ref
            for b in detect_stumps(model, f):
                if b[4] < 0.4 or (b[3] - b[1]) < 1.5 * (b[2] - b[0]):
                    continue
                base = np.array([(b[0] + b[2]) / 2, b[3]])
                raw.append(base)
                if m_ok:
                    mapped.append(motion.map_points(i, base[None])[0])
            if m_ok:
                zooms.append(motion.zoom(i))
        if len(mapped) < 5:
            rows.append({"clip": video.name, "frames": len(frames), "mapped_frames": len(motion.to_ref), "note": "too few stumps detections"})
            continue
        r_raw, r_map = landmark_residuals(np.array(raw)), landmark_residuals(np.array(mapped))
        rows.append({"clip": video.name, "frames": len(frames), "mapped_frames": len(motion.to_ref),
                     "stumps_detections": len(mapped), "zoom_range": [float(min(zooms)), float(max(zooms))],
                     "raw_median_px": float(np.median(r_raw)), "raw_p90_px": float(np.percentile(r_raw, 90)),
                     "stabilised_median_px": float(np.median(r_map)), "stabilised_p90_px": float(np.percentile(r_map, 90)),
                     "warnings": motion.warnings})
        print(json.dumps(rows[-1]), flush=True)
    ok = [r for r in rows if "stabilised_median_px" in r]
    med = lambda k: float(np.nanmedian([r[k] for r in ok])) if ok else None  # noqa: E731
    res = {"clips": len(rows), "evaluated": len(ok),
           "raw_unmeasurable_clips": sum(1 for r in ok if np.isnan(r["raw_median_px"])),
           "clip_median_of": {k: med(k) for k in ("raw_median_px", "raw_p90_px", "stabilised_median_px", "stabilised_p90_px")},
           "rows": rows,
           "note": "Residual = distance of a detected stumps base from its static landmark (raw image positions vs. positions "
                   "mapped into the reference frame); includes stumps-detector noise. Landmarks with < 5 detections are dropped "
                   "as false detections. Note: with the raw positions a landmark spread by camera motion may split into several "
                   "groups, which understates the raw error."}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
