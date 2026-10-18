"""Run the cricket-synth stump-plane benchmark and write outputs/reports/synth_benchmark.json.

    python scripts/evaluate_trajectory_synth.py --limit 100 --noise 1.0 --impact-y 1.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.evaluation.synth_benchmark import evaluate_clip, load_clip, summarize  # noqa: E402
from ml.lbw.engine import LbwConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/hf-cricket-synth/out/labels"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--noise", type=float, default=1.0, help="Gaussian pixel noise (sigma) added to GT centres")
    ap.add_argument("--impact-y", type=float, default=1.5, help="virtual pad distance in front of stumps (m)")
    ap.add_argument("--rig", default="front_on_stump")
    ap.add_argument("--radius-noise", type=float, default=0.5, help="noise (px) on apparent radius; <0 disables the radius cue")
    ap.add_argument("--inflation", type=float, default=2.0, help="LBW covariance inflation factor")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N matching clips (for held-out splits)")
    ap.add_argument("--swing-sigma", type=float, default=2.0, help="lateral swing prior sigma (m/s^2); 0 disables the swing term")
    ap.add_argument("--no-prior", action="store_true", help="disable the bounce speed-retention prior")
    ap.add_argument("--drag-cd-prior", default=None, help="'mean,sd' of Cd: fit drag per delivery (default: fixed Cd 0.4)")
    ap.add_argument("--swing-post-sigma", type=float, default=0.0, help="separate post-bounce lateral acceleration prior sigma (0 = off)")
    ap.add_argument("--reproject-rig", default=None,
                    help="re-project the same simulated trajectories through cameras of this manifest rig (e.g. side_on)")
    ap.add_argument("--out", default=str(ROOT / "outputs/reports/synth_benchmark.json"))
    args = ap.parse_args()

    fit_kwargs = {}
    if args.drag_cd_prior:
        mu, sd = (float(v) for v in args.drag_cd_prior.split(","))
        fit_kwargs["drag_cd_prior"] = (mu, sd)
    if args.swing_post_sigma > 0:
        fit_kwargs["swing_post_prior_sigma"] = args.swing_post_sigma
    manifests = {}
    for mf in (Path(args.data).parents[1] / "manifests").glob("*.jsonl"):
        for line in mf.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                manifests[row["clip_id"]] = row
    other_cams = []
    if args.reproject_rig:
        from ml.evaluation.synth_benchmark import manifest_camera

        other_cams = [manifest_camera(r) for r in manifests.values() if r["camera"] == args.reproject_rig]
        if not other_cams:
            print(f"No manifest cameras for rig {args.reproject_rig}")
            return 1
    files = sorted(Path(args.data).glob("*.json"))
    if not files:
        print(f"No label files under {args.data}. Run scripts/download_datasets.py --only hf-cricket-synth")
        return 1
    results = []
    t0 = time.perf_counter()
    skipped = 0
    lbw_cfg = LbwConfig(n_samples=500, covariance_inflation=args.inflation)
    for f in files:
        d = load_clip(f)
        if args.rig and d["camera"]["rig"] != args.rig:
            continue
        if skipped < args.offset:
            skipped += 1
            continue
        cam = None
        if other_cams:
            from ml.evaluation.synth_benchmark import reproject_clip

            cam = other_cams[len(results) % len(other_cams)]
            d = reproject_clip(d, cam, args.reproject_rig)
        results.append(evaluate_clip(d, impact_y=args.impact_y, pixel_noise=args.noise,
                                     use_radius=args.radius_noise >= 0, radius_noise=max(args.radius_noise, 0.0),
                                     use_prior=not args.no_prior, lbw_cfg=lbw_cfg,
                                     swing_sigma=args.swing_sigma or None, fit_kwargs=fit_kwargs,
                                     true_cd=manifests.get(d["clip_id"], {}).get("cd"), camera=cam))
        if len(results) >= args.limit:
            break
    summary = summarize(results)
    summary.update(dataset="Ranvin36/cricket-synth (synthetic)", pixel_noise_sigma=args.noise, impact_y_m=args.impact_y,
                   rig=args.rig, radius_cue=args.radius_noise >= 0, radius_noise_px=args.radius_noise,
                   bounce_prior=not args.no_prior, covariance_inflation=args.inflation, swing_sigma=args.swing_sigma,
                   clip_offset=args.offset, runtime_s=time.perf_counter() - t0, drag_cd_prior=args.drag_cd_prior,
                   swing_post_sigma=args.swing_post_sigma, reproject_rig=args.reproject_rig,
                   caveat="Synthetic data with a known camera. Measures trajectory/projection layers only; "
                          "not evidence of real-world accuracy.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "clips": [asdict(r) for r in results]}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
