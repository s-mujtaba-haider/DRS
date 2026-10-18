"""Ball-labelling workflow: frame plans, detector pre-labels and export of human labels as a YOLO set.

Labels are ordinary ``Annotation`` rows (label "ball": add/move with a radius = ball present,
mark_absent = no ball visible), so corrections made while reviewing tracks count too.
Pre-labels are suggestions only and are never stored unless a person accepts them.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.api.deps import db, get_or_404, settings, to_dict
from backend.app.core.config import ROOT, Settings
from backend.app.database.models import Annotation, Video

router = APIRouter(prefix="/api/labeling", tags=["labeling"])
_DET: dict = {}
_DET_LOCK = threading.Lock()
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _labels(s: Session, video_id: str) -> dict[int, dict]:
    from ml.datasets.ui_labels import latest_labels

    return latest_labels([to_dict(a) for a in s.query(Annotation).filter(Annotation.video_id == video_id).all()])


@router.get("/videos/{video_id}/plan")
def plan(video_id: str, step: int = 5, start: int = 0, end: int | None = None, s: Session = Depends(db)):
    """Frames to label (every `step` frames) and which already carry a human label."""
    v = get_or_404(s, Video, video_id)
    if not 1 <= step <= 500:
        raise HTTPException(422, "step must be 1-500")
    n = v.meta.get("frame_count_decoded") or v.meta.get("frame_count_declared") or 0
    stop = min(end if end is not None else n, n) if n else (end or 0)
    frames = list(range(max(0, start), stop, step))
    lab = _labels(s, video_id)
    status = {f: ("ball" if a["action"] in ("add", "move") else "no_ball") for f, a in lab.items()}
    return {"video_id": video_id, "frames": frames, "labelled": {str(f): status[f] for f in frames if f in status},
            "all_labelled": {"ball": sum(1 for x in status.values() if x == "ball"),
                             "no_ball": sum(1 for x in status.values() if x == "no_ball")}}


def _detector():
    from ml.pipeline import CRICKET_WEIGHTS

    with _DET_LOCK:
        if "det" not in _DET:
            if not Path(CRICKET_WEIGHTS).is_file():
                raise HTTPException(503, "Cricket ball detector weights are not installed; label without suggestions")
            from ml.detection.yolo_detector import YoloBallDetector

            _DET["det"] = YoloBallDetector(str(CRICKET_WEIGHTS), conf=0.05)
            _DET["id"] = Path(CRICKET_WEIGHTS).name
        return _DET["det"], _DET["id"]


@router.get("/videos/{video_id}/prelabel/{frame_index}")
def prelabel(video_id: str, frame_index: int, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    """Detector suggestions for one frame (to accept or correct; never stored automatically)."""
    from ml.detection.base import FrameInfo
    from ml.video.ingest import iter_frames

    v = get_or_404(s, Video, video_id)
    rec = next(iter_frames(cfg.videos_dir / v.stored_name, start_frame=frame_index, max_frames=1), None)
    if rec is None:
        raise HTTPException(404, "Frame index beyond end of video")
    det, model_id = _detector()
    with _DET_LOCK:
        dets = det.detect_batch([rec.image], [FrameInfo(rec.index, rec.timestamp_ms, rec.timestamp_source)])[0]
    dets = sorted(dets, key=lambda d: -d.confidence)[:5]
    return {"model_id": model_id, "frame_index": frame_index, "note": "Model suggestions; confirm or correct before saving",
            "suggestions": [{"x": d.center[0], "y": d.center[1], "radius_px": max(2.0, ((d.bbox[2] - d.bbox[0]) + (d.bbox[3] - d.bbox[1])) / 4),
                             "confidence": d.confidence} for d in dets]}


class ExportRequest(BaseModel):
    video_ids: list[str] = Field(min_length=1, max_length=200)
    name: str = Field(description="output set name under data/annotations/")
    default_box_px: float = Field(12.0, gt=1, le=200)


@router.post("/export", status_code=201)
def export(body: ExportRequest, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    """Write human labels to data/annotations/<name> (YOLO layout + provenance, one sequence per video)."""
    from ml.datasets.ui_labels import export_ui_labels

    if not _NAME.match(body.name):
        raise HTTPException(422, "name: 1-64 letters, digits, '_' or '-'")
    out = cfg.annotations_dir / body.name
    results = []
    for vid in body.video_ids:
        v = get_or_404(s, Video, vid)
        anns = [to_dict(a) for a in s.query(Annotation).filter(Annotation.video_id == vid).all()]
        seq = re.sub(r"[^A-Za-z0-9_-]", "_", Path(v.original_filename).stem)[:40] + f"_{v.id[:6]}"
        results.append(export_ui_labels(cfg.videos_dir / v.stored_name, seq, v.meta["width"], v.meta["height"], anns, out,
                                        body.default_box_px))
    total = {k: sum(r[k] for r in results) for k in ("positives", "negatives", "frames")}
    shown = out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out)
    return {"path": shown, "videos": results, **total,
            "next_step": f"python scripts/annotate_video.py build --sets {shown} --name <dataset>"}
