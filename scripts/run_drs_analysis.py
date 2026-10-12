"""Full offline analysis: tracking -> (calibration) -> experimental LBW, using configs/*.yaml.

    python scripts/run_drs_analysis.py VIDEO --camera camera.json --out outputs/delivery1 \
        --handedness right --impact-frame 212

Without --camera only tracking and 2D trajectory are produced (no 3D / LBW: a calibrated
camera is required). Results are written as JSON next to the tracking outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.analysis import run_lbw  # noqa: E402
from ml.calibration.camera import CameraModel  # noqa: E402
from ml.lbw.engine import LbwConfig  # noqa: E402
from ml.lbw.rules import LbwFacts, RulesConfig  # noqa: E402
from ml.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def load_yaml(name: str) -> dict:
    p = ROOT / "configs" / name
    return yaml.safe_load(p.read_text()) if p.is_file() else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", help="CameraModel JSON (scripts/calibrate_cameras.py or exported from the UI)")
    ap.add_argument("--impact-frame", type=int)
    ap.add_argument("--handedness", choices=["right", "left"])
    ap.add_argument("--no-ball", choices=["yes", "no"])
    ap.add_argument("--bat-first", choices=["yes", "no"])
    ap.add_argument("--shot-offered", choices=["yes", "no"])
    a = ap.parse_args()

    det_cfg = load_yaml("detection.yaml")
    trk_cfg = load_yaml("tracking.yaml")
    cfg = PipelineConfig(**{k: v for k, v in det_cfg.items() if k in PipelineConfig.__dataclass_fields__}, tracker=trk_cfg)
    summary = run_pipeline(a.video, a.out, cfg, progress=lambda s, f: print(f"\r{s:>8s} {f * 100:5.1f}%", end="", flush=True))
    print()
    result = {"tracking_summary": summary}
    if a.camera:
        cam = CameraModel.load(a.camera)
        lbw_yaml = load_yaml("lbw.yaml")
        rules = RulesConfig(**lbw_yaml.get("rules", {}))
        lcfg = LbwConfig(**{k: v for k, v in lbw_yaml.items() if k in LbwConfig.model_fields and k != "rules"}, rules=rules)
        yn = {"yes": True, "no": False, None: None}
        facts = LbwFacts(batter_handedness=a.handedness, no_ball=yn[a.no_ball], ball_touched_bat_first=yn[a.bat_first],
                         genuine_attempt_to_play=yn[a.shot_offered])
        impact_t = None
        if a.impact_frame is not None:
            tr = json.loads((Path(a.out) / "tracking.json").read_text())
            ts = {p["frame_index"]: p["timestamp_ms"] for t in tr["tracks"] for p in t["points"]}
            impact_t = ts.get(a.impact_frame, None)
            impact_t = None if impact_t is None else impact_t / 1000
        res, overlay = run_lbw(Path(a.out), cam, facts, lcfg, impact_t, [])
        result["lbw"] = {**res.model_dump(mode="json"), "overlay": overlay}
        print(json.dumps({k: result["lbw"][k] for k in ("verdict", "p_hit", "rules_summary", "reasons", "warnings")}, indent=2))
    else:
        print("No --camera given: LBW/3D analysis skipped (requires calibration).")
    (Path(a.out) / "drs_analysis.json").write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
