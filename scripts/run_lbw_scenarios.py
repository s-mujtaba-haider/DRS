"""End-to-end LBW scenario test through the running API (synthetic deliveries with known outcomes).

Each scenario renders a synthetic delivery video, then goes through the real workflow: upload,
tracking job, calibration (exact intrinsics, landmark clicks from the true camera), LBW analysis,
and an LBW review whose recommendation applies the ICC T20I protocol. With ``--multiview`` a second,
side-on 100 fps camera is rendered for the same delivery and the multi-camera LBW endpoint is used
(hardware sync asserted, as for a triggered rig). The ground truth comes from
the generating flight model, so every row can be checked. Synthetic: not evidence of real accuracy.

    python scripts/run_lbw_scenarios.py [--api http://127.0.0.1:8100] [--fps 50]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.calibration.camera import look_at_camera  # noqa: E402
from ml.calibration.pitch_geometry import BallSpec, WicketSpec, standard_landmarks  # noqa: E402
from ml.evaluation.synthetic import SyntheticDelivery, make_delivery, write_synthetic_video  # noqa: E402
from ml.trajectory.ballistic import DEFAULT_DRAG_C  # noqa: E402

SIDE_POINTS = ["striker_off_stump_base_rh", "striker_leg_stump_base_rh", "striker_off_stump_top_rh", "striker_leg_stump_top_rh",
               "striker_middle_stump_top", "striker_middle_stump_base", "striker_popping_crease_x_plus_1m32",
               "striker_popping_crease_x_minus_1m32", "striker_popping_crease_center"]


def side_view(d: SyntheticDelivery, fps: float) -> SyntheticDelivery:
    """The same flight seen by a side-on camera (square of the batter) at its own frame rate."""
    cam = look_at_camera("side_on", np.array([14.0, 3.0, 1.5]), np.array([0.0, 3.0, 0.5]), 1500.0, 1280, 720)
    times = np.arange(0.0, d.impact_time + 1e-9, 1.0 / fps)
    xyz = d.model.position(times)
    return SyntheticDelivery(model=d.model, times=times, xyz=xyz, camera=cam, uv=cam.project(xyz), impact_time=d.impact_time)


def track(B, H, path: Path, name: str):
    with open(path, "rb") as fh:
        v = requests.post(f"{B}/api/videos", headers=H, files={"file": (name, fh, "video/mp4")}).json()
    job = requests.post(f"{B}/api/analysis/jobs", headers=H, json={"video_id": v["id"], "detector": "motion"}).json()
    for _ in range(240):
        j = requests.get(f"{B}/api/analysis/jobs/{job['id']}").json()
        if j["status"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return v, j
        time.sleep(0.5)
    return v, j


def calibrate(B, H, v, cam, names):
    lm = standard_landmarks()
    pts = [{"name": n, "image": cam.project(np.array([lm[n]]))[0].tolist()} for n in names]
    return requests.post(f"{B}/api/calibration", headers=H, json={"video_id": v["id"], "points": pts,
                                                                  "intrinsics": {"K": cam.K, "validated": True}}).json()

CAL_POINTS = ["striker_off_stump_base_rh", "striker_leg_stump_base_rh", "striker_off_stump_top_rh", "striker_leg_stump_top_rh",
              "striker_middle_stump_top", "striker_popping_crease_x_plus_1m32", "striker_popping_crease_x_minus_1m32",
              "striker_middle_stump_base", "striker_bowling_crease_right", "striker_bowling_crease_left"]

# name: (make_delivery kwargs, on-field call)
SCENARIOS = {
    "hitting middle": (dict(line_x=0.0, end_x=0.0), "NOT_OUT"),
    "missing off (wide)": (dict(line_x=0.05, end_x=0.32), "OUT"),
    "missing leg": (dict(line_x=0.0, end_x=-0.30), "OUT"),
    "going over the top": (dict(line_x=0.0, end_x=0.0, post_bounce_vz=7.5), "OUT"),
    "clipping off stump (marginal)": (dict(line_x=0.05, end_x=0.13), "NOT_OUT"),
    "pitched outside leg": (dict(line_x=-0.30, end_x=-0.02), "OUT"),
}


def truth(d) -> dict:
    """Where the generating model crosses the stump plane, and the zone tests at that point."""
    w, r = WicketSpec(), BallSpec().radius_m
    ts = np.arange(0, 1.5, 1e-4)
    p = d.model.position(ts)
    k = int(np.argmax(p[:, 1] <= 0.0))
    x, z = float(p[k, 0]), float(p[k, 2])
    hw, top = w.half_width_m, w.top_m
    any_part = abs(x) <= hw + r and z - r <= top
    centre = abs(x) <= hw and z <= top
    pb = d.model.position(np.array([d.model.bounce_time()]))[0] if d.model.bounce else None
    pitched_leg = pb is not None and pb[0] < -(hw + r)  # right-hander: leg side is -x
    return {"x_at_stumps_cm": round(100 * x, 1), "z_at_stumps_cm": round(100 * z, 1), "would_hit": bool(any_part),
            "centre_in_zone": bool(centre), "pitched_outside_leg": bool(pitched_leg)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8100")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--pin", default=None)
    ap.add_argument("--multiview", action="store_true", help="add a side-on 100 fps camera and use multi-camera LBW")
    ap.add_argument("--no-drag", action="store_true", help="drag-free flight (unrealistic; the engine assumes real drag)")
    a = ap.parse_args()
    B = a.api.rstrip("/")
    H = {"X-DRS-PIN": a.pin} if a.pin else {}
    lm = standard_landmarks()
    m = requests.post(f"{B}/api/matches", headers=H, json={"name": f"LBW scenario test {time.strftime('%H:%M')}", "team_a": "Lions",
                                                           "team_b": "Tigers", "protocol_id": "icc_t20i_2025", "batting_team": "Lions",
                                                           "officials": {"third_umpire": "TV Umpire"}}).json()
    rows, tmp = [], Path(tempfile.mkdtemp())
    for name, (kw, on_field) in SCENARIOS.items():
        d = make_delivery(fps=a.fps, drag_c=0.0 if a.no_drag else DEFAULT_DRAG_C, **kw)
        gt = truth(d)
        path = tmp / f"lbw_{name.split()[0]}_{len(rows)}.mp4"
        write_synthetic_video(str(path), d, fps=a.fps, lead_frames=8, tail_frames=5)
        with open(path, "rb") as fh:
            v = requests.post(f"{B}/api/videos", headers=H, files={"file": (f"LBW test - {name}.mp4", fh, "video/mp4")}).json()
        job = requests.post(f"{B}/api/analysis/jobs", headers=H, json={"video_id": v["id"], "detector": "motion"}).json()
        for _ in range(240):
            j = requests.get(f"{B}/api/analysis/jobs/{job['id']}").json()
            if j["status"] in ("succeeded", "failed", "cancelled", "interrupted"):
                break
            time.sleep(0.5)
        if j["status"] != "succeeded":
            rows.append({"scenario": name, "error": j.get("error")})
            continue
        cam = d.camera
        pts = [{"name": n, "image": cam.project(np.array([lm[n]]))[0].tolist()} for n in CAL_POINTS]
        cal = requests.post(f"{B}/api/calibration", headers=H, json={"video_id": v["id"], "points": pts,
                                                                     "intrinsics": {"K": cam.K, "validated": True}}).json()
        facts = {"batter_handedness": "right", "no_ball": False, "ball_touched_bat_first": False,
                 "genuine_attempt_to_play": True, "intercepted_by_person_not_bat_hand": True}
        if a.multiview:
            sd = side_view(d, 100.0)
            spath = tmp / f"side_{len(rows)}.mp4"
            # 16 lead frames at 100 fps = 8 at 50 fps: both clips start 160 ms before release (common clock)
            write_synthetic_video(str(spath), sd, fps=100.0, lead_frames=16, tail_frames=10)
            sv, sj = track(B, H, spath, f"LBW test - {name} (side-on).mp4")
            scal = calibrate(B, H, sv, sd.camera, SIDE_POINTS)
            lbw = requests.post(f"{B}/api/lbw/multiview", headers=H, json={
                "cameras": [{"job_id": job["id"], "calibration_id": cal["id"], "sync_uncertainty_ms": 0.0},
                            {"job_id": sj["id"], "calibration_id": scal["id"], "sync_uncertainty_ms": 0.0}],
                "facts": facts, "hardware_sync": True}).json()
            if "result" not in lbw:
                rows.append({"scenario": name, "error": lbw})
                continue
            v_ids = [v["id"], sv["id"]]
        else:
            lbw = requests.post(f"{B}/api/lbw/analyze", headers=H, json={"job_id": job["id"], "calibration_id": cal["id"], "facts": facts}).json()
            v_ids = [v["id"]]
        res = lbw["result"]
        rv = requests.post(f"{B}/api/matches/{m['id']}/reviews", headers=H,
                           json={"requested_by": "umpire", "decision_type": "lbw", "on_field": on_field, "over_ball": f"{len(rows) + 1}.1",
                                 "video_ids": v_ids}).json()
        requests.post(f"{B}/api/reviews/{rv['id']}/evidence", headers=H, json={"type": "lbw", "id": lbw["id"]})
        rec = requests.post(f"{B}/api/reviews/{rv['id']}/recommend", headers=H).json()["recommendation"]
        wp = res.get("wicket_projection") or {}
        cr = wp.get("crossing") or {}
        rows.append({
            "scenario": name, "truth": gt, "on_field": on_field,
            "verdict": res["verdict"], "p_hit": res.get("p_hit"), "rules_summary": res.get("rules_summary"),
            "pred_x_cm": round(100 * cr["mean"][0], 1) if cr.get("mean") else None,
            "pred_z_cm": round(100 * cr["mean"][1], 1) if cr.get("mean") else None,
            "recommendation": rec["outcome"], "suggested": rec["final_decision"],
            "components": {c["name"]: c["status"] for c in rec.get("components", [])},
            "review_url": f"http://localhost:3100/reviews/{rv['id']}", "session_url": f"http://localhost:3100/sessions/{job['id']}",
        })
        print(f"done: {name}", flush=True)
    out = ROOT / "outputs" / "reports" / ("lbw_scenarios_e2e_multiview.json" if a.multiview else "lbw_scenarios_e2e.json")
    out.write_text(json.dumps(rows, indent=1))
    print()
    print(f"{'scenario':32} {'truth':28} {'verdict':18} {'P(hit)':>6}  {'pred vs true (x,z cm)':26} {'recommendation':26} on-field")
    for r in rows:
        if "error" in r:
            print(f"{r['scenario']:32} ERROR {r['error']}")
            continue
        t = r["truth"]
        tt = ("HIT" if t["would_hit"] else "MISS") + (" (centre in)" if t["centre_in_zone"] else " (centre out)" if t["would_hit"] else "")
        if t["pitched_outside_leg"]:
            tt += ", pitched leg"
        p = f"{r['p_hit']:.2f}" if r["p_hit"] is not None else "n/a"
        pv = f"({r['pred_x_cm']},{r['pred_z_cm']}) vs ({t['x_at_stumps_cm']},{t['z_at_stumps_cm']})"
        print(f"{r['scenario']:32} {tt:28} {r['verdict']:18} {p:>6}  {pv:26} {r['recommendation'] + ' -> ' + r['suggested']:26} {r['on_field']}")
    print(f"\nmatch: http://localhost:3100/matches/{m['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
