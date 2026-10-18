"""Multi-camera LBW and clip-group helpers.

A *clip group* is the set of videos exported together from live capture (``meta.capture.clip_group``);
they share one clock, so their tracking results can be triangulated after time alignment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.api.deps import db, get_or_404, jobs, settings, to_dict
from backend.app.core.config import Settings
from backend.app.database.models import Annotation, Calibration, Job, LbwAnalysis, Video
from backend.app.services.analysis import AnalysisInputError, observations_for_lbw, project_overlay
from backend.app.workers.jobs import JobManager
from ml.calibration.camera import CameraModel
from ml.lbw.engine import LbwConfig
from ml.lbw.multiview import MultiViewConfig, ViewObservations, analyze_lbw_multiview
from ml.lbw.rules import LbwFacts

router = APIRouter(prefix="/api", tags=["multiview"])
ROOT = Path(__file__).resolve().parents[3]


@lru_cache
def _multiview_defaults() -> dict:
    p = ROOT / "configs" / "lbw.yaml"
    try:
        return (yaml.safe_load(p.read_text()) or {}).get("multiview") or {}
    except (OSError, yaml.YAMLError):
        return {}


# ------------------------------------------------------------------ clip groups
def _group_videos(s: Session, group: str) -> list[Video]:
    vids = [v for v in s.query(Video).order_by(Video.created_at).all()
            if ((v.meta or {}).get("capture") or {}).get("clip_group") == group]
    if not vids:
        raise HTTPException(404, f"Clip group {group} not found")
    return vids


@router.get("/clip-groups/{group}")
def clip_group(group: str, s: Session = Depends(db)):
    vids = _group_videos(s, group)
    out = []
    for v in vids:
        js = s.query(Job).filter(Job.video_id == v.id).order_by(Job.created_at.desc()).all()
        out.append({"video": to_dict(v), "camera_id": v.meta["capture"].get("camera_id"), "jobs": [to_dict(j) for j in js]})
    return {"clip_group": group, "videos": out}


class GroupTrack(BaseModel):
    detector: str = Field("auto", pattern="^(auto|cricket|coco_hybrid|coco|motion)$")
    conf: float = Field(0.15, ge=0.01, le=0.99)
    imgsz: int = Field(1280, ge=320, le=3840)
    device: str = Field("auto", pattern=r"^(auto|cpu|\d)$")
    render_video: bool = True
    force: bool = Field(False, description="queue even if the video already has an active or succeeded job")


@router.post("/clip-groups/{group}/track", status_code=201)
def track_clip_group(group: str, body: GroupTrack | None = None, s: Session = Depends(db), mgr: JobManager = Depends(jobs),
                     cfg: Settings = Depends(settings)):
    """Queue one tracking job per camera clip in the group (jobs run one at a time on the GPU)."""
    body = body or GroupTrack()
    queued, skipped = [], []
    for v in _group_videos(s, group):
        existing = s.query(Job).filter(Job.video_id == v.id, Job.status.in_(["queued", "running", "succeeded"])).first()
        if existing is not None and not body.force:
            skipped.append({"video_id": v.id, "job_id": existing.id, "status": existing.status})
            continue
        conf = {"detector": body.detector, "conf": body.conf, "imgsz": body.imgsz, "tile_size": None, "batch_size": 8,
                "device": body.device, "half": True, "frame_step": 1, "start_frame": 0, "end_frame": None,
                "max_frames": cfg.max_frames_per_job, "render_video": body.render_video}
        j = Job(video_id=v.id, config=conf)
        s.add(j)
        s.commit()
        mgr.submit(j.id)
        queued.append(to_dict(j))
    return {"clip_group": group, "queued": queued, "skipped": skipped}


# --------------------------------------------------------------- multi-view LBW
class CameraInput(BaseModel):
    job_id: str
    calibration_id: str
    extra_offset_ms: float = Field(0.0, description="added to the clip's capture offset, e.g. from /api/sync")
    sync_uncertainty_ms: float | None = Field(None, ge=0, description="default: the clip's capture sync uncertainty")


class MultiViewLbwRequest(BaseModel):
    cameras: list[CameraInput] = Field(min_length=2, max_length=8)
    impact_time_ms: float | None = Field(None, description="impact time on the shared clock (ms)")
    facts: LbwFacts = Field(default_factory=LbwFacts)
    decision_confidence: float = Field(0.95, ge=0.5, lt=1.0)
    require_validated_calibration_for_definite: bool = True
    hardware_sync: bool = Field(False, description="assert that the cameras are hardware-synchronised (trigger/PTP)")
    max_sync_position_error_m: float | None = Field(None, gt=0, le=1.0)


@router.post("/lbw/multiview", status_code=201)
def lbw_multiview(body: MultiViewLbwRequest, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    """LBW from 2+ cameras: triangulated 3D track with synchronisation uncertainty propagated.

    Stored as an LbwAnalysis (job = first camera's job), so it can be attached to a review as type "lbw".
    """
    if len({c.job_id for c in body.cameras}) != len(body.cameras):
        raise HTTPException(422, "each camera needs its own tracking job")
    views, resolved, first = [], [], None
    seen_ids: set[str] = set()
    for i, c in enumerate(body.cameras):
        job = get_or_404(s, Job, c.job_id)
        if job.status != "succeeded":
            raise HTTPException(409, f"Tracking job {job.id} has not succeeded")
        # A calibration made on another clip from the same fixed camera is allowed.
        cal = get_or_404(s, Calibration, c.calibration_id)
        video = s.get(Video, job.video_id)
        cam = CameraModel.model_validate(cal.camera)
        if cam.camera_id in seen_ids:
            cam.camera_id = f"{cam.camera_id}_{i}"
        seen_ids.add(cam.camera_id)
        capture = (video.meta or {}).get("capture") or {}
        base_offset = capture.get("time_offset_ms")
        offset = (base_offset if base_offset is not None else cam.time_offset_ms) + c.extra_offset_ms
        unc = c.sync_uncertainty_ms
        if unc is None:
            unc = capture.get("sync_uncertainty_ms", cam.sync_uncertainty_ms)
        if unc is None:
            raise HTTPException(422, f"Sync uncertainty unknown for camera {cam.camera_id} (job {job.id}); "
                                     "supply sync_uncertainty_ms (it cannot be assumed to be zero)")
        anns = [to_dict(a) for a in s.query(Annotation).filter(Annotation.video_id == job.video_id, Annotation.label == "ball").all()]
        try:
            t, uv, _, notes, frames = observations_for_lbw(cfg.jobs_dir / job.id, anns)
        except AnalysisInputError as exc:
            raise HTTPException(422, f"Camera {cam.camera_id}: {exc}") from exc
        views.append(ViewObservations(camera=cam, t_ms=(t * 1000.0).tolist(), uv=uv.tolist(), frame_index=frames.tolist(),
                                      time_offset_ms=float(offset), sync_uncertainty_ms=float(unc),
                                      hardware_sync=body.hardware_sync))
        resolved.append({"camera_id": cam.camera_id, "job_id": job.id, "video_id": video.id, "calibration_id": cal.id,
                         "time_offset_ms": float(offset), "sync_uncertainty_ms": float(unc),
                         "offset_source": "capture" if base_offset is not None else "calibration",
                         "clip_group": capture.get("clip_group"), "n_observations": len(t), "notes": notes})
        if first is None:
            first = (job, cal, cam, t, uv, float(offset))
    groups = {r["clip_group"] for r in resolved}
    warnings = []
    if len(groups) > 1 or None in groups:
        warnings.append("Cameras are not all from one capture clip group: their clocks are only related through the "
                        "supplied offsets")
    mv = MultiViewConfig(**{k: v for k, v in _multiview_defaults().items() if k in MultiViewConfig.model_fields})
    if body.max_sync_position_error_m is not None:
        mv.max_sync_position_error_m = body.max_sync_position_error_m
    lcfg = LbwConfig(decision_confidence=body.decision_confidence,
                     require_validated_calibration_for_definite=body.require_validated_calibration_for_definite)
    try:
        result, diag, _ = analyze_lbw_multiview(views, body.facts, lcfg, mv, body.impact_time_ms)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    result.warnings.extend(warnings)
    job, cal, cam, t, uv, off = first
    # Overlay for the first camera's footage: observation times shifted to the shared clock used by the fit.
    overlay = project_overlay(cam, result, t + off / 1000.0, uv)
    rec = LbwAnalysis(job_id=job.id, calibration_id=cal.id,
                      request={**body.model_dump(mode="json"), "multiview": True, "resolved_cameras": resolved},
                      result={**result.model_dump(mode="json"), "overlay": overlay,
                              "multiview": diag.model_dump(mode="json") if diag else None})
    s.add(rec)
    s.commit()
    return to_dict(rec)
