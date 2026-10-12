"""Run detection + tracking + 2D trajectory on a video and write outputs.

    python scripts/run_tracking.py VIDEO --out outputs/run1 --detector auto --imgsz 1280 --batch 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector", default="auto", choices=["auto", "cricket", "coco_hybrid", "coco", "motion"])
    ap.add_argument("--weights")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", type=int, default=None, help="tile size for small-object inference")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    cfg = PipelineConfig(detector=a.detector, weights=a.weights, conf=a.conf, imgsz=a.imgsz, tile_size=a.tile,
                         batch_size=a.batch, device=a.device, half=not a.no_half, frame_step=a.step,
                         start_frame=a.start, end_frame=a.end, max_frames=a.max_frames, render_video=not a.no_video)

    def progress(stage: str, frac: float) -> None:
        print(f"\r{stage:>8s} {frac * 100:5.1f}%", end="", flush=True)

    summary = run_pipeline(a.video, a.out, cfg, progress=progress)
    print()
    print(json.dumps({k: summary[k] for k in ("detector", "frames_processed", "num_tracks", "primary_track",
                                             "trajectory_2d", "throughput_fps", "gpu_peak_mb", "warnings")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
