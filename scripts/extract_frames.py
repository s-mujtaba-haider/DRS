"""Extract frames (with original indices and timestamps) from a video.

    python scripts/extract_frames.py VIDEO --out data/processed/frames/clip1 --step 2 --start 100 --end 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.video.ingest import extract_frames, probe_video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--format", default="jpg", choices=["jpg", "png"])
    a = ap.parse_args()
    meta = probe_video(a.video)
    print(meta.model_dump_json(indent=2))
    idx = extract_frames(a.video, a.out, step=a.step, start_frame=a.start, end_frame=a.end, image_format=a.format)
    print(f"Wrote frames and {idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
