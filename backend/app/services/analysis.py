"""Glue between stored job outputs and the ML analysis modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.calibration.camera import CameraModel
from ml.calibration.pitch_geometry import BallSpec, WicketSpec, standard_landmarks
from ml.common.schemas import TrackingResult
from ml.lbw.engine import LbwConfig, LbwResult, LbwVerdict, analyze_lbw
from ml.lbw.rules import LbwFacts
from ml.trajectory.ballistic import InsufficientObservations, fit_to_pixels


class AnalysisInputError(ValueError):
    pass


def load_tracking(job_dir: Path) -> TrackingResult:
    p = job_dir / "tracking.json"
    if not p.is_file():
        raise AnalysisInputError("Tracking results not found for this job")
    return TrackingResult.model_validate_json(p.read_text())


def load_detections_index(job_dir: Path) -> dict[tuple[int, float, float], dict]:
    """Map (frame, cx, cy) -> detection dict, to recover box sizes for observed points."""
    p = job_dir / "detections.json"
    out = {}
    if p.is_file():
        for fd in json.loads(p.read_text()):
            for d in fd["detections"]:
                out[(d["frame_index"], round(d["center"][0], 2), round(d["center"][1], 2))] = d
    return out


def observations_for_lbw(job_dir: Path, annotations: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Primary-track observations with manual corrections applied (corrections win, originals untouched on disk)."""
    tracking = load_tracking(job_dir)
    primary = next((t for t in tracking.tracks if t.is_primary), None)
    notes: list[str] = []
    if primary is None:
        raise AnalysisInputError("No primary ball track: run tracking or add manual ball annotations")
    if not primary.quality.reliable:
        notes.append("Primary track flagged unreliable: " + "; ".join(primary.quality.reliability_issues))
    dets = load_detections_index(job_dir)
    obs: dict[int, tuple[float, float, float, float]] = {}
    for p in primary.points:
        if p.kind != "observed":
            continue
        d = dets.get((p.frame_index, round(p.x, 2), round(p.y, 2)))
        r = np.nan
        if d and d.get("source") == "model":
            # Only learned-detector boxes outline the ball; motion-difference masks do not, so
            # their size is not a valid depth cue (measured bias ~+2 px on synthetic footage).
            x1, y1, x2, y2 = d["bbox"]
            r = min(x2 - x1, y2 - y1) / 2.0  # min side: motion blur elongates the other
        obs[p.frame_index] = (p.timestamp_ms / 1000.0, p.x, p.y, r)
    frame_ts = {p.frame_index: p.timestamp_ms for p in primary.points}
    n_manual = 0
    for a in sorted(annotations, key=lambda a: a["created_at"]):
        f = a["frame_index"]
        if a["action"] in ("reject", "mark_absent"):
            obs.pop(f, None)
            n_manual += 1
        elif a["action"] in ("add", "move"):
            ts = a["data"].get("timestamp_ms", frame_ts.get(f))
            if ts is None:
                continue
            obs[f] = (ts / 1000.0, float(a["data"]["x"]), float(a["data"]["y"]), float(a["data"].get("radius_px") or np.nan))
            n_manual += 1
    if n_manual:
        notes.append(f"{n_manual} manual correction(s) applied")
    frames = sorted(obs)
    arr = np.array([obs[f] for f in frames]) if frames else np.zeros((0, 4))
    return arr[:, 0], arr[:, 1:3], arr[:, 3], notes


def run_lbw(job_dir: Path, cam: CameraModel, facts: LbwFacts, cfg: LbwConfig, impact_time_s: float | None,
            annotations: list[dict], use_radius: bool = True) -> tuple[LbwResult, dict]:
    t, uv, radius, notes = observations_for_lbw(job_dir, annotations)
    overlay: dict = {"notes": notes}
    if impact_time_s is not None:
        keep = t <= impact_time_s + 1e-6
        t, uv, radius = t[keep], uv[keep], radius[keep]
    try:
        fit = fit_to_pixels(cam, t, uv, sigma_px=2.0, ball_radius=BallSpec().radius_m,
                            radius_px=radius if use_radius else None, radius_sigma_px=1.0)
    except InsufficientObservations as exc:
        res = analyze_lbw(None, facts, cfg)
        res.reasons.append(str(exc))
        return res, overlay
    res = analyze_lbw(fit, facts, cfg, impact_time_s=impact_time_s, calibration_validated=cam.validated)
    res.warnings.extend(notes)
    overlay.update(project_overlay(cam, res, t, uv))
    return res, overlay


def project_overlay(cam: CameraModel, res: LbwResult, t, uv) -> dict:
    """Image-space geometry so the UI can draw the analysis on top of the footage."""

    def proj(points):
        if not points:
            return []
        P = np.asarray(points, float)
        front = cam.is_in_front(P)
        out = cam.project(P)
        return [[float(u), float(v)] if ok else None for (u, v), ok in zip(out, front)]

    lm = standard_landmarks()
    w = WicketSpec()
    hw, top = w.half_width_m, w.top_m
    wicket = [[-hw, 0, 0], [hw, 0, 0], [hw, 0, top], [-hw, 0, top]]
    o = {
        "observations_px": uv.tolist(),
        "observation_times_s": list(map(float, t)),
        "fitted_path_px": proj([p[1:] for p in res.observed_path_fit]),
        "predicted_path_px": proj([p[1:] for p in res.predicted_path]),
        "wicket_outline_px": proj(wicket),
        "striker_crease_px": proj([lm["striker_popping_crease_x_minus_1m32"], lm["striker_popping_crease_x_plus_1m32"]]),
    }
    if res.pitching_point:
        o["pitching_point_px"] = proj([res.pitching_point.mean])[0]
    if res.impact_point:
        o["impact_point_px"] = proj([res.impact_point.mean])[0]
    wp = res.wicket_projection
    if wp and wp.crossing:
        x, z = wp.crossing.mean
        sx, sz = wp.crossing.std
        o["stump_crossing_px"] = proj([[x, 0, z]])[0]
        ell = [[x + 2 * sx * np.cos(a), 0, z + 2 * sz * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 24)]
        o["stump_crossing_2sigma_px"] = proj(ell)
    return o


VERDICT_STYLE = {
    LbwVerdict.PROJECTED_TO_HIT.value: "hit",
    LbwVerdict.PROJECTED_TO_MISS.value: "miss",
    LbwVerdict.INCONCLUSIVE.value: "uncertain",
    LbwVerdict.INSUFFICIENT_DATA.value: "uncertain",
}
