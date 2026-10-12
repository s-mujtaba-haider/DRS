"""Calibration, LBW analysis, multi-view reconstruction and contact analysis endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from backend.app.api.deps import db, get_or_404, settings, to_dict
from backend.app.core.config import Settings
from backend.app.database.models import Annotation, Calibration, Job, LbwAnalysis, Video
from backend.app.services.analysis import AnalysisInputError, run_lbw
from ml.calibration.camera import CalibrationError, CameraModel, estimate_pose
from ml.calibration.pitch_geometry import standard_landmarks
from ml.lbw.engine import LbwConfig
from ml.lbw.rules import LbwFacts
from ml.reconstruction.triangulation import CameraObservation, associate_and_triangulate

router = APIRouter(prefix="/api", tags=["analysis"])


# ------------------------------------------------------------------ calibration
@router.get("/calibration/landmarks")
def landmarks():
    """Named pitch landmarks (metres, pitch frame) usable as calibration points."""
    return {"frame": "origin at striker's middle-stump base; +y towards bowler; +z up; +x = off side of a right-hander",
            "landmarks": standard_landmarks()}


class PointPair(BaseModel):
    name: str | None = None
    world: tuple[float, float, float] | None = None
    image: tuple[float, float]

    @model_validator(mode="after")
    def _one_of(self):
        if self.world is None and self.name is None:
            raise ValueError("each point needs a landmark name or explicit world coordinates")
        return self


class Intrinsics(BaseModel):
    K: list[list[float]]
    dist: list[float] = Field(default_factory=lambda: [0.0] * 5)
    validated: bool = False


class CalibrationCreate(BaseModel):
    video_id: str | None = None
    camera_id: str = "main"
    image_width: int | None = Field(None, gt=0)
    image_height: int | None = Field(None, gt=0)
    points: list[PointPair] = Field(min_length=4)
    intrinsics: Intrinsics | None = None
    time_offset_ms: float = 0.0
    sync_uncertainty_ms: float | None = None


@router.post("/calibration", status_code=201)
def create_calibration(body: CalibrationCreate, s: Session = Depends(db)):
    w, h = body.image_width, body.image_height
    if body.video_id:
        v = get_or_404(s, Video, body.video_id)
        w, h = w or v.meta["width"], h or v.meta["height"]
    if not (w and h):
        raise HTTPException(422, "image size required (or give video_id)")
    lm = standard_landmarks()
    world, image = [], []
    for p in body.points:
        if p.world is not None:
            world.append(p.world)
        elif p.name in lm:
            world.append(lm[p.name])
        else:
            raise HTTPException(422, f"Unknown landmark '{p.name}'")
        image.append(p.image)
    try:
        cam = estimate_pose(body.camera_id, np.array(world), np.array(image), (w, h),
                            K=np.array(body.intrinsics.K) if body.intrinsics else None,
                            dist=np.array(body.intrinsics.dist) if body.intrinsics else None,
                            intrinsics_validated=bool(body.intrinsics and body.intrinsics.validated))
    except CalibrationError as exc:
        raise HTTPException(422, str(exc)) from exc
    cam.time_offset_ms, cam.sync_uncertainty_ms = body.time_offset_ms, body.sync_uncertainty_ms
    per_point = np.linalg.norm(cam.project(np.array(world)) - np.array(image), axis=1)
    c = Calibration(video_id=body.video_id, camera=cam.model_dump(mode="json"),
                    correspondences={**body.model_dump(mode="json"), "per_point_error_px": per_point.round(3).tolist()})
    s.add(c)
    s.commit()
    return to_dict(c)


@router.post("/calibration/import", status_code=201)
def import_calibration(camera: CameraModel, video_id: str | None = None, s: Session = Depends(db)):
    camera.calibration_type = "imported"
    camera.warnings.append("Imported calibration: provenance not verified by this system")
    c = Calibration(video_id=video_id, camera=camera.model_dump(mode="json"), correspondences={"imported": True})
    s.add(c)
    s.commit()
    return to_dict(c)


@router.get("/calibration")
def list_calibrations(video_id: str | None = None, s: Session = Depends(db)):
    q = s.query(Calibration)
    if video_id:
        q = q.filter(Calibration.video_id == video_id)
    return [to_dict(c) for c in q.order_by(Calibration.created_at.desc()).all()]


@router.get("/calibration/{cal_id}")
def get_calibration(cal_id: str, s: Session = Depends(db)):
    return to_dict(get_or_404(s, Calibration, cal_id))


@router.get("/calibration/{cal_id}/export")
def export_calibration(cal_id: str, s: Session = Depends(db)):
    c = get_or_404(s, Calibration, cal_id)
    return Response(CameraModel.model_validate(c.camera).model_dump_json(indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="camera_{cal_id}.json"'})


# -------------------------------------------------------------------------- LBW
class LbwRequest(BaseModel):
    job_id: str
    calibration_id: str
    impact_frame: int | None = Field(None, ge=0)
    facts: LbwFacts = Field(default_factory=LbwFacts)
    decision_confidence: float = Field(0.95, ge=0.5, lt=1.0)
    require_validated_calibration_for_definite: bool = True
    use_ball_size_depth_cue: bool = True


@router.post("/lbw/analyze", status_code=201)
def lbw_analyze(body: LbwRequest, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    job = get_or_404(s, Job, body.job_id)
    if job.status != "succeeded":
        raise HTTPException(409, "Tracking job has not succeeded")
    cal = get_or_404(s, Calibration, body.calibration_id)
    cam = CameraModel.model_validate(cal.camera)
    impact_t = None
    if body.impact_frame is not None:
        import json

        tracking = json.loads((cfg.jobs_dir / job.id / "tracking.json").read_text())
        ts = {p["frame_index"]: p["timestamp_ms"] for t in tracking["tracks"] for p in t["points"]}
        if body.impact_frame not in ts:
            raise HTTPException(422, "impact_frame is outside the tracked span; choose a frame on the ball track")
        impact_t = ts[body.impact_frame] / 1000.0
    anns = [to_dict(a) for a in s.query(Annotation).filter(Annotation.video_id == job.video_id, Annotation.label == "ball").all()]
    lbw_cfg = LbwConfig(decision_confidence=body.decision_confidence,
                        require_validated_calibration_for_definite=body.require_validated_calibration_for_definite)
    try:
        result, overlay = run_lbw(cfg.jobs_dir / job.id, cam, body.facts, lbw_cfg, impact_t, anns,
                                  use_radius=body.use_ball_size_depth_cue)
    except AnalysisInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    rec = LbwAnalysis(job_id=job.id, calibration_id=cal.id, request=body.model_dump(mode="json"),
                      result={**result.model_dump(mode="json"), "overlay": overlay})
    s.add(rec)
    s.commit()
    return to_dict(rec)


@router.get("/lbw")
def list_lbw(job_id: str | None = None, s: Session = Depends(db)):
    q = s.query(LbwAnalysis)
    if job_id:
        q = q.filter(LbwAnalysis.job_id == job_id)
    return [to_dict(a) for a in q.order_by(LbwAnalysis.created_at.desc()).all()]


@router.get("/lbw/{analysis_id}")
def get_lbw(analysis_id: str, s: Session = Depends(db)):
    return to_dict(get_or_404(s, LbwAnalysis, analysis_id))


# ------------------------------------------------------------- reconstruction
class TriangulateRequest(BaseModel):
    calibration_ids: list[str] = Field(min_length=2)
    observations: list[CameraObservation] = Field(min_length=2)
    sync_tolerance_ms: float = Field(2.0, gt=0, le=100)
    max_reprojection_px: float = Field(5.0, gt=0)
    fit_flight_model: bool = True


@router.post("/reconstruction/triangulate")
def triangulate(body: TriangulateRequest, s: Session = Depends(db)):
    cams = [CameraModel.model_validate(get_or_404(s, Calibration, cid).camera) for cid in body.calibration_ids]
    if len({c.camera_id for c in cams}) != len(cams):
        raise HTTPException(422, "camera_id values must be unique across the selected calibrations")
    try:
        res = associate_and_triangulate(cams, body.observations, body.sync_tolerance_ms, body.max_reprojection_px)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    out = res.model_dump(mode="json")
    if body.fit_flight_model and len(res.points) >= 6:
        from ml.trajectory.ballistic import InsufficientObservations, fit_to_points

        t = np.array([p.timestamp_ms / 1000 for p in res.points])
        X = np.array([p.xyz for p in res.points])
        sig = np.array([np.sqrt(max(np.trace(np.array(p.cov)) / 3, 1e-8)) for p in res.points])
        try:
            out["flight_fit"] = fit_to_points(t, X, sig).summary()
        except InsufficientObservations as exc:
            out["flight_fit"] = {"status": "insufficient_data", "reasons": [str(exc)]}
    return out


# ------------------------------------------------------------------- contact
class ContactRequest(BaseModel):
    job_id: str
    window_frames: int = Field(6, ge=3, le=50)


@router.post("/contact/analyze")
def contact_analyze(body: ContactRequest, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    from backend.app.services.analysis import load_tracking
    from ml.contact_analysis.deflection import analyze_deflection

    job = get_or_404(s, Job, body.job_id)
    if job.status != "succeeded":
        raise HTTPException(409, "Tracking job has not succeeded")
    v = s.get(Video, job.video_id)
    tracking = load_tracking(cfg.jobs_dir / job.id)
    primary = next((t for t in tracking.tracks if t.is_primary), None)
    if primary is None:
        raise HTTPException(422, "No primary ball track")
    fps = (v.meta.get("nominal_fps") or 0) / (job.config.get("frame_step") or 1)
    return analyze_deflection(primary, fps, body.window_frames).model_dump(mode="json")
