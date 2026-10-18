"""Run-out, stumping, boundary and wicket-break evidence analyses (stored as DecisionAnalysis rows).

Each analysis is computed from operator-marked (or model-proposed, operator-reviewed) observations
and stored with its full request, so a review's evidence can be reproduced exactly.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError, model_validator
from sqlalchemy.orm import Session

from backend.app.api.deps import db, get_or_404, settings, to_dict
from backend.app.core.config import Settings
from backend.app.database.models import Calibration, DecisionAnalysis, Video

router = APIRouter(prefix="/api", tags=["decisions"])


@router.get("/decision-analyses")
def list_analyses(video_id: str | None = None, kind: str | None = None, s: Session = Depends(db)):
    q = s.query(DecisionAnalysis)
    if video_id:
        q = q.filter(DecisionAnalysis.video_id == video_id)
    if kind:
        q = q.filter(DecisionAnalysis.kind == kind)
    return [to_dict(a) for a in q.order_by(DecisionAnalysis.created_at.desc()).all()]


@router.get("/decision-analyses/{analysis_id}")
def get_analysis(analysis_id: str, s: Session = Depends(db)):
    return to_dict(get_or_404(s, DecisionAnalysis, analysis_id))


def _store(s: Session, kind: str, video_id: str | None, calibration_id: str | None, request: dict, result: dict) -> dict:
    a = DecisionAnalysis(kind=kind, video_id=video_id, calibration_id=calibration_id, request=request, result=result)
    s.add(a)
    s.commit()
    return to_dict(a)


def _camera(s: Session, calibration_id: str | None):
    from ml.calibration.camera import CameraModel

    if calibration_id is None:
        return None
    return CameraModel.model_validate(get_or_404(s, Calibration, calibration_id).camera)


def _video_size(v: Video) -> tuple[int, int]:
    return int(v.meta["width"]), int(v.meta["height"])


# ------------------------------------------------------------- wicket break
class WicketBreakRequest(BaseModel):
    video_id: str
    roi: tuple[int, int, int, int] = Field(description="x1, y1, x2, y2 around stumps and bails, image pixels")
    start_frame: int = Field(0, ge=0)
    end_frame: int | None = Field(None, ge=1)
    persist_frames: int = Field(3, ge=1, le=20)

    @model_validator(mode="after")
    def _check(self):
        x1, y1, x2, y2 = self.roi
        if x2 - x1 < 8 or y2 - y1 < 8:
            raise ValueError("roi must be at least 8x8 px with x2 > x1 and y2 > y1")
        if self.end_frame is not None and self.end_frame <= self.start_frame:
            raise ValueError("end_frame must exceed start_frame")
        return self


MAX_BREAK_SCAN_FRAMES = 1500


@router.post("/wicket-break/detect", status_code=201)
def wicket_break(body: WicketBreakRequest, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    """Automatic wicket-break *candidate*. Never confirmed: the operator confirms bail removal (Law 29.1)."""
    from ml.runout.events import detect_wicket_break_in_video

    v = get_or_404(s, Video, body.video_id)
    w, h = _video_size(v)
    x1, y1, x2, y2 = body.roi
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise HTTPException(422, "roi outside the frame")
    end = min(body.end_frame or body.start_frame + MAX_BREAK_SCAN_FRAMES, body.start_frame + MAX_BREAK_SCAN_FRAMES)
    det = detect_wicket_break_in_video(cfg.videos_dir / v.stored_name, body.roi, start_frame=body.start_frame,
                                       end_frame=end, persist_frames=body.persist_frames)
    return _store(s, "wicket_break", v.id, None, body.model_dump(mode="json"), det.model_dump(mode="json"))


# --------------------------------------------------------- run-out/stumping
class CreaseSpec(BaseModel):
    source: Literal["manual", "calibration"]
    p1: tuple[float, float] | None = None
    p2: tuple[float, float] | None = None
    behind_point: tuple[float, float] | None = Field(None, description="manual: any image point on the stump side")
    calibration_id: str | None = None
    end: Literal["striker", "bowler"] | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.source == "manual" and None in (self.p1, self.p2, self.behind_point):
            raise ValueError("manual crease needs p1, p2 (back edge of the crease marking) and behind_point")
        if self.source == "calibration" and (self.calibration_id is None or self.end is None):
            raise ValueError("calibration crease needs calibration_id and end")
        return self


class CreaseRequest(BaseModel):
    video_id: str
    dismissal: Literal["run_out", "stumping"]
    crease: CreaseSpec
    observations: list[dict] = Field(min_length=1, max_length=2000)
    wicket_break: dict | None = None
    facts: dict = Field(default_factory=dict)
    decision_confidence: float = Field(0.95, ge=0.5, lt=1.0)


@router.post("/runout/analyze", status_code=201)
def runout_analyze(body: CreaseRequest, s: Session = Depends(db)):
    """Run-out (Law 38) or stumping (Law 39) crease analysis from marked observations."""
    from ml.runout.crease import CreaseGeometryError, crease_from_camera, crease_from_points
    from ml.runout.engine import BatterObservation, CreaseConfig, WicketBreak
    from ml.runout.rules import DismissalFacts, analyze_runout, analyze_stumping

    v = get_or_404(s, Video, body.video_id)
    try:
        obs = [BatterObservation.model_validate(o) for o in body.observations]
        wb = WicketBreak.model_validate(body.wicket_break) if body.wicket_break else None
        facts = DismissalFacts.model_validate(body.facts)
        cam = _camera(s, body.crease.calibration_id) if body.crease.source == "calibration" else None
        line = (crease_from_camera(cam, body.crease.end) if cam is not None
                else crease_from_points(body.crease.p1, body.crease.p2, body.crease.behind_point, body.crease.end))
    except (ValidationError, CreaseGeometryError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    w, h = _video_size(v)
    if any(o.point and not (0 <= o.point[0] <= w and 0 <= o.point[1] <= h) for o in obs):
        raise HTTPException(422, "observation point outside the frame")
    fn = analyze_runout if body.dismissal == "run_out" else analyze_stumping
    res = fn(line, sorted(obs, key=lambda o: o.timestamp_ms), wb, facts,
             CreaseConfig(decision_confidence=body.decision_confidence))
    out = res.model_dump(mode="json")
    out["crease_line_px"] = [list(line.p1), list(line.p2)]
    return _store(s, body.dismissal, v.id, body.crease.calibration_id, body.model_dump(mode="json"), out)


# ----------------------------------------------------------------- boundary
class BoundaryRequest(BaseModel):
    video_id: str | None = None
    analysis: Literal["boundary", "catch"] = "boundary"
    rope: dict
    calibration_id: str | None = None
    events: list[dict] = Field(min_length=1, max_length=500)
    facts: dict = Field(default_factory=dict)
    decision_confidence: float = Field(0.95, ge=0.5, lt=1.0)


@router.post("/boundary/analyze", status_code=201)
def boundary_analyze(body: BoundaryRequest, s: Session = Depends(db)):
    """Boundary (Law 19) or catch-near-boundary (Laws 19.5 / 33) analysis of an observed event sequence."""
    from pydantic import TypeAdapter

    from ml.boundary.events import BoundaryEvent
    from ml.boundary.geometry import BoundaryRope
    from ml.boundary.rules import BoundaryConfig, BoundaryFacts, analyze_boundary, analyze_catch_near_boundary

    if body.video_id:
        get_or_404(s, Video, body.video_id)
    try:
        rope_d = dict(body.rope)
        cam = _camera(s, body.calibration_id)
        if cam is not None:
            rope_d["camera"] = cam
        rope = BoundaryRope.model_validate(rope_d)
        events = TypeAdapter(list[BoundaryEvent]).validate_python(body.events)
        facts = BoundaryFacts.model_validate(body.facts)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    events.sort(key=lambda e: e.timestamp_ms)
    fn = analyze_boundary if body.analysis == "boundary" else analyze_catch_near_boundary
    res = fn(events, rope, facts, BoundaryConfig(decision_confidence=body.decision_confidence))
    return _store(s, body.analysis, body.video_id, body.calibration_id, body.model_dump(mode="json"),
                  res.model_dump(mode="json"))
