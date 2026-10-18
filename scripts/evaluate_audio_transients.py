"""Run audio transient detection over real broadcast delivery clips (no contact labels exist).

This measures how often the detector fires on ordinary broadcast audio (commentary, crowd, stump mic),
a proxy for its false-alarm burden. It cannot measure edge-detection accuracy.

    python scripts/evaluate_audio_transients.py [--glob "data/raw/hf-cricket-delivery-videos/videos/*.mp4"] [--limit 60]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.contact_analysis.audio import analyze_video_audio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/raw/hf-cricket-delivery-videos/videos/*.mp4")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="outputs/reports/audio_transients_broadcast.json")
    a = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / a.glob)))[: a.limit]
    rows, t0 = [], time.perf_counter()
    for f in files:
        r = analyze_video_audio(f)
        dur = None
        if r.window_ms:
            dur = (r.window_ms[1] - r.window_ms[0]) / 1000
        rows.append({"file": Path(f).name, "status": r.status, "duration_s": dur, "n_transients": len(r.transients),
                     "per_second": (len(r.transients) / dur) if dur else None, "noise_floor_db": r.noise_floor_db,
                     "snr_db": [t.snr_db for t in r.transients], "highpass_hz": r.highpass_hz})
    n = np.array([r["n_transients"] for r in rows if r["status"] != "NO_AUDIO"])
    rate = np.array([r["per_second"] for r in rows if r["per_second"] is not None])
    summary = {"clips": len(rows), "with_audio": int(len(n)), "clips_with_transient": int((n > 0).sum()),
               "transients_per_clip_median": float(np.median(n)) if len(n) else None,
               "transients_per_clip_p90": float(np.percentile(n, 90)) if len(n) else None,
               "transients_per_second_median": float(np.median(rate)) if len(rate) else None,
               "seconds": round(time.perf_counter() - t0, 1),
               "caveat": "No contact labels: this is the detector's firing rate on broadcast audio, not edge accuracy."}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "clips": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
