"""Estimate the time offset between two camera recordings.

    python scripts/estimate_sync.py audio cam_a.mp4 cam_b.mp4 [--max-offset-ms 2000]
    python scripts/estimate_sync.py events --a 1240,3880.5 --b 1201.2,3841.7 --fps-a 50 --fps-b 100
    python scripts/estimate_sync.py audio a.mp4 b.mp4 --apply-to camera_b.json --out camera_b_synced.json

Output convention: t_a = t_b + offset_ms (camera A is the reference clock).
Events are timestamps (ms, as in iter_frames / the review UI) of the first frame
in which each matched event (bail break, bat-ball contact, bounce) is visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.calibration.camera import CameraModel  # noqa: E402
from ml.synchronization import apply_sync, audio_offset, event_offset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audio")
    a.add_argument("video_a")
    a.add_argument("video_b")
    a.add_argument("--sample-rate", type=int, default=16000)
    a.add_argument("--max-offset-ms", type=float, default=2000.0)
    a.add_argument("--max-duration-s", type=float, default=120.0)
    e = sub.add_parser("events")
    e.add_argument("--a", required=True, help="comma-separated event times in camera A (ms)")
    e.add_argument("--b", required=True, help="matched event times in camera B (ms)")
    e.add_argument("--fps-a", type=float, required=True)
    e.add_argument("--fps-b", type=float, required=True)
    e.add_argument("--mode", choices=["first_seen", "nearest"], default="first_seen")
    for p in (a, e):
        p.add_argument("--apply-to", help="camera B calibration JSON (CameraModel) to update")
        p.add_argument("--out", help="where to write the updated camera JSON")
    args = ap.parse_args()
    if args.cmd == "audio":
        res = audio_offset(args.video_a, args.video_b, args.sample_rate, args.max_offset_ms, args.max_duration_s)
    else:
        res = event_offset([float(x) for x in args.a.split(",")], [float(x) for x in args.b.split(",")],
                           args.fps_a, args.fps_b, args.mode)
    print(res.model_dump_json(indent=2))
    if args.apply_to:
        cam = apply_sync(CameraModel.load(args.apply_to), res)
        out = args.out or args.apply_to.replace(".json", "_synced.json")
        cam.save(out)
        print(f"wrote {out} (time_offset_ms={cam.time_offset_ms:.3f}, sync_uncertainty_ms={cam.sync_uncertainty_ms:.3f})")


if __name__ == "__main__":
    main()
