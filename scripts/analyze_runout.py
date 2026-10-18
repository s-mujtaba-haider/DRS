"""Run-out / stumping analysis CLI (experimental).

Detect a wicket-break candidate in a clip (ROI around stumps and bails, top band = bails):
    python scripts/analyze_runout.py detect --video clip.mp4 --roi 820 300 900 420 [--start-frame 0 --end-frame 200]

Analyse crease status with operator observations (JSON list of BatterObservation:
{"frame_index", "timestamp_ms", "point": [u, v], "grounded": true|false|null}):
    python scripts/analyze_runout.py analyze --observations obs.json \
        (--camera camera.json --end striker | --crease u1 v1 u2 v2 bu bv) \
        (--break 140 141 | --video clip.mp4 --roi ...) [--facts facts.json] [--dismissal run_out|stumped]

``--break`` gives the last intact and first broken frame indices; their timestamps are taken from the
observations file, so include those frames there. Without it the wicket break is detected automatically
and reported as an unconfirmed candidate. Output is JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.calibration.camera import CameraModel  # noqa: E402
from ml.runout.crease import crease_from_camera, crease_from_points  # noqa: E402
from ml.runout.engine import BatterObservation, WicketBreak  # noqa: E402
from ml.runout.events import detect_wicket_break_in_video  # noqa: E402
from ml.runout.rules import DismissalFacts, analyze_runout, analyze_stumping  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    a = sub.add_parser("analyze")
    for p in (d, a):
        p.add_argument("--video")
        p.add_argument("--roi", nargs=4, type=float)
        p.add_argument("--start-frame", dest="start", type=int, default=0)
        p.add_argument("--end-frame", dest="end_frame", type=int)
    a.add_argument("--observations", required=True)
    a.add_argument("--camera")
    a.add_argument("--end", dest="crease_end", choices=["striker", "bowler"])
    a.add_argument("--crease", nargs=6, type=float, metavar=("U1", "V1", "U2", "V2", "BU", "BV"))
    a.add_argument("--break", dest="brk", nargs=2, type=int, metavar=("LAST_INTACT", "FIRST_BROKEN"))
    a.add_argument("--confirmed", action="store_true", help="operator confirmed a bail completely removed")
    a.add_argument("--facts")
    a.add_argument("--dismissal", choices=["run_out", "stumped"], default="run_out")
    args = ap.parse_args(argv)

    if args.cmd == "detect":
        if not (args.video and args.roi):
            ap.error("detect needs --video and --roi")
        print(detect_wicket_break_in_video(args.video, args.roi, args.start, args.end_frame).model_dump_json(indent=2))
        return 0

    obs = [BatterObservation.model_validate(o) for o in json.loads(Path(args.observations).read_text())]
    if args.camera:
        if not args.crease_end:
            ap.error("--camera needs --end striker|bowler")
        line = crease_from_camera(CameraModel.load(args.camera), args.crease_end)
    elif args.crease:
        u1, v1, u2, v2, bu, bv = args.crease
        line = crease_from_points((u1, v1), (u2, v2), (bu, bv), args.crease_end)
    else:
        ap.error("give --camera/--end or --crease")
    if args.brk:
        ts = {o.frame_index: o.timestamp_ms for o in obs}
        if not all(f in ts for f in args.brk):
            ap.error("--break frames must appear in the observations file (for their timestamps)")
        wb = WicketBreak(last_intact_frame=args.brk[0], last_intact_ms=ts[args.brk[0]], first_broken_frame=args.brk[1],
                         first_broken_ms=ts[args.brk[1]], source="manual", confirmed=args.confirmed)
    elif args.video and args.roi:
        wb = WicketBreak.from_detection(detect_wicket_break_in_video(args.video, args.roi, args.start, args.end_frame))
    else:
        wb = None
    facts = DismissalFacts.model_validate(json.loads(Path(args.facts).read_text())) if args.facts else DismissalFacts()
    fn = analyze_stumping if args.dismissal == "stumped" else analyze_runout
    print(fn(line, obs, wb, facts).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
