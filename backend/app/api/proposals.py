"""Model proposals of batter ground-contact points for the run-out / stumping workspace.

Proposals are suggestions from a generic pose model; the operator confirms or corrects them
before they become observations for ``/api/runout/analyze``. Nothing is stored.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from backend.app.api.deps import db, get_or_404, settings
from backend.app.core.config import Settings
from backend.app.database.models import Video

router = APIRouter(prefix="/api", tags=["proposals"])
_MODEL: dict = {}
_LOCK = threading.Lock()


def get_pose_model():
    """Lazily loaded pose model (module-level so tests can monkeypatch it)."""
    from ml.runout.proposals import UltralyticsPose

    with _LOCK:
        if "model" not in _MODEL:
            _MODEL["model"] = UltralyticsPose()
        return _MODEL["model"]


class CreaseLinePx(BaseModel):
    p1: tuple[float, float]
    p2: tuple[float, float]


class ProposalRequest(BaseModel):
    video_id: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=1, description="exclusive")
    crease: CreaseLinePx | None = None
    roi: tuple[float, float, float, float] | None = Field(None, description="x1, y1, x2, y2 around the batter's end")

    @model_validator(mode="after")
    def _check(self):
        from ml.runout.proposals import MAX_FRAMES

        if self.end_frame <= self.start_frame:
            raise ValueError("end_frame must exceed start_frame")
        if self.end_frame - self.start_frame > MAX_FRAMES:
            raise ValueError(f"at most {MAX_FRAMES} frames per request")
        if self.roi is not None and (self.roi[2] <= self.roi[0] or self.roi[3] <= self.roi[1]):
            raise ValueError("roi must satisfy x2 > x1 and y2 > y1")
        if self.crease is None and self.roi is None:
            raise ValueError("give the crease line or a region around the batter's end")
        if self.crease is not None and self.crease.p1 == self.crease.p2:
            raise ValueError("crease points must differ")
        return self


@router.post("/runout/proposals")
def runout_proposals(body: ProposalRequest, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    """Propose ankle / sole points of the batter in each frame, with a heuristic grounded hint."""
    from ml.runout.proposals import PoseUnavailableError, propose_in_video

    v = get_or_404(s, Video, body.video_id)
    w, h = int(v.meta["width"]), int(v.meta["height"])
    pts = ([body.crease.p1, body.crease.p2] if body.crease else []) + (
        [body.roi[:2], body.roi[2:]] if body.roi else [])
    if any(not (0 <= x <= w and 0 <= y <= h) for x, y in pts):
        raise HTTPException(422, "crease or roi outside the frame")
    try:
        model = get_pose_model()
    except PoseUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    res = propose_in_video(cfg.videos_dir / v.stored_name, model, body.start_frame, body.end_frame,
                           crease=(body.crease.p1, body.crease.p2) if body.crease else None, roi=body.roi)
    return res.model_dump(mode="json")
