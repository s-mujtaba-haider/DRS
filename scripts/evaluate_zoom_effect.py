"""How much does an uncompensated broadcast zoom bias the LBW stump-plane projection? (synthetic)

cricket-synth deliveries (exact camera) are re-observed by a camera zooming about its principal
point by ``rate`` per frame, relative to a calibration frame in the middle of the observed flight.
Three cases go through the standard benchmark (ml/evaluation/synth_benchmark.evaluate_clip):

* ``ignored``: zoomed pixels and radii fitted with the calibration-frame camera (the old pipeline);
* ``compensated``: observations mapped back into the calibration frame, with extra Gaussian
  noise ``--residual-px`` standing in for the measured stabilisation residual on real clips;
* ``baseline``: no zoom at all.

    python scripts/evaluate_zoom_effect.py --offset 120 --limit 120 --rates 0.005,0.01 \
        --out outputs/reports/zoom_effect_synth.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.evaluation.synth_benchmark import HALF_L, evaluate_clip, load_clip, summarize  # noqa: E402
from ml.lbw.engine import LbwConfig  # noqa: E402


def zoomed(d: dict, rate: float, impact_y: float, residual_px: float, compensate: bool, rng) -> dict:
    d = copy.deepcopy(d)
    intr = d["camera"]["intrinsics"]
    c = np.array([intr["cx"], intr["cy"]])
    obs = [k for k, f in enumerate(d["frames"]) if f["ball_in_scene"] and f["world_m"] is not None and f["visible"]
           and f["centre_px"] is not None and f["world_m"][1] + HALF_L > impact_y]
    if not obs:
        return d
    ref = obs[len(obs) // 2]
    for k in obs:
        f = d["frames"][k]
        z = (1.0 + rate) ** (k - ref)
        if compensate:
            f["centre_px"] = (np.asarray(f["centre_px"], float) + rng.normal(0, residual_px, 2)).tolist()
        else:
            f["centre_px"] = (c + z * (np.asarray(f["centre_px"], float) - c)).tolist()
            if f.get("radius_px"):
                f["radius_px"] = f["radius_px"] * z
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/raw/hf-cricket-synth/out/labels"))
    ap.add_argument("--rig", default="front_on_stump")
    ap.add_argument("--offset", type=int, default=120)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--rates", default="0.005,0.01")
    ap.add_argument("--residual-px", type=float, default=1.5)
    ap.add_argument("--out")
    a = ap.parse_args()
    files = sorted(Path(a.data).glob("*.json"))
    clips = [d for d in (load_clip(p) for p in files) if d["camera"]["rig"] == a.rig][a.offset:a.offset + a.limit]
    cfg = LbwConfig(n_samples=500, covariance_inflation=2.0)
    report = {"clips": len(clips), "rig": a.rig, "residual_px": a.residual_px, "results": {}}
    rng = np.random.default_rng(0)
    cases = [("baseline", 0.0, False)] + [(f"{m}_{r}", float(r), m == "compensated")
                                           for r in a.rates.split(",") for m in ("ignored", "compensated")]
    for name, rate, comp in cases:
        res = [evaluate_clip(zoomed(d, rate, 1.5, a.residual_px, comp, rng) if rate else d, lbw_cfg=cfg, seed=i,
                             swing_sigma=2.0) for i, d in enumerate(clips)]
        s = summarize(res)
        report["results"][name] = {k: s.get(k) for k in ("num_with_projection", "stump_plane_error_m", "inconclusive_rate",
                                                         "definite_verdicts", "false_hit", "false_miss",
                                                         "interval_90_empirical_coverage")}
        print(name, json.dumps(report["results"][name]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
