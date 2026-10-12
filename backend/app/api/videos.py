"""Video upload, metadata, streaming and manual annotations."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.api.deps import db, get_or_404, settings, to_dict
from backend.app.core.config import Settings
from backend.app.database.models import Annotation, Video
from ml.video.ingest import VideoDecodeError, probe_video

router = APIRouter(prefix="/api/videos", tags=["videos"])
CHUNK = 1 << 20


@router.post("", status_code=201)
async def upload_video(file: UploadFile = File(...), s: Session = Depends(db), cfg: Settings = Depends(settings)):
    name = Path(file.filename or "").name  # never trust client paths
    ext = Path(name).suffix.lower()
    if ext not in cfg.allowed_extensions:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {', '.join(cfg.allowed_extensions)}")
    stored = f"{uuid.uuid4().hex}{ext}"
    dest = cfg.videos_dir / stored
    limit = cfg.max_upload_mb * 1024 * 1024
    h, size = hashlib.sha256(), 0
    try:
        with dest.open("wb") as fh:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File exceeds the {cfg.max_upload_mb} MB upload limit")
                h.update(chunk)
                fh.write(chunk)
        if size == 0:
            raise HTTPException(400, "Uploaded file is empty")
        try:
            meta = probe_video(dest)
        except VideoDecodeError as exc:
            raise HTTPException(422, f"Not a decodable video: {exc}") from exc
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    meta.path_name = name
    v = Video(original_filename=name[:255], stored_name=stored, size_bytes=size, sha256=h.hexdigest(),
              meta=meta.model_dump(mode="json"))
    s.add(v)
    s.commit()
    return to_dict(v)


@router.get("")
def list_videos(s: Session = Depends(db)):
    return [to_dict(v) for v in s.query(Video).order_by(Video.created_at.desc()).all()]


@router.get("/{video_id}")
def get_video(video_id: str, s: Session = Depends(db)):
    return to_dict(get_or_404(s, Video, video_id))


@router.get("/{video_id}/file")
def video_file(video_id: str, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    v = get_or_404(s, Video, video_id)
    path = (cfg.videos_dir / v.stored_name).resolve()
    if path.parent != cfg.videos_dir.resolve() or not path.is_file():
        raise HTTPException(404, "Video file missing")
    return FileResponse(path, media_type="video/mp4" if path.suffix in (".mp4", ".m4v") else None)


@router.get("/{video_id}/frame/{frame_index}")
def video_frame(video_id: str, frame_index: int, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    """Exact decoded frame as PNG (frame-accurate reference for annotation and calibration)."""
    import cv2

    from ml.video.ingest import iter_frames

    v = get_or_404(s, Video, video_id)
    if frame_index < 0:
        raise HTTPException(422, "frame_index must be >= 0")
    rec = next(iter_frames(cfg.videos_dir / v.stored_name, start_frame=frame_index, max_frames=1), None)
    if rec is None:
        raise HTTPException(404, "Frame index beyond end of video")
    ok, png = cv2.imencode(".png", rec.image)
    return Response(png.tobytes(), media_type="image/png",
                    headers={"X-Frame-Timestamp-Ms": f"{rec.timestamp_ms:.3f}", "X-Timestamp-Source": rec.timestamp_source.value})


@router.delete("/{video_id}", status_code=204)
def delete_video(video_id: str, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    from backend.app.database.models import Job

    v = get_or_404(s, Video, video_id)
    if s.query(Job).filter(Job.video_id == video_id, Job.status.in_(["queued", "running"])).count():
        raise HTTPException(409, "Video has active jobs")
    p = cfg.videos_dir / v.stored_name
    s.delete(v)
    s.commit()
    if p.is_file():
        os.remove(p)


class AnnotationIn(BaseModel):
    frame_index: int = Field(ge=0)
    action: Literal["add", "move", "reject", "mark_absent"]
    label: str = "ball"
    job_id: str | None = None
    x: float | None = None
    y: float | None = None
    radius_px: float | None = Field(None, gt=0)
    timestamp_ms: float | None = None
    note: str | None = Field(None, max_length=500)


@router.post("/{video_id}/annotations", status_code=201)
def add_annotation(video_id: str, body: AnnotationIn, s: Session = Depends(db)):
    v = get_or_404(s, Video, video_id)
    if body.action in ("add", "move") and (body.x is None or body.y is None):
        raise HTTPException(422, "x and y are required for add/move")
    w, h = v.meta["width"], v.meta["height"]
    if body.x is not None and not (0 <= body.x <= w and 0 <= (body.y or 0) <= h):
        raise HTTPException(422, "Point outside the frame")
    a = Annotation(video_id=video_id, job_id=body.job_id, frame_index=body.frame_index, label=body.label,
                   action=body.action, data=body.model_dump(exclude={"frame_index", "action", "label", "job_id"}, exclude_none=True))
    s.add(a)
    s.commit()
    return to_dict(a)


@router.get("/{video_id}/annotations")
def list_annotations(video_id: str, s: Session = Depends(db)):
    get_or_404(s, Video, video_id)
    rows = s.query(Annotation).filter(Annotation.video_id == video_id).order_by(Annotation.created_at).all()
    return [to_dict(a) for a in rows]


@router.delete("/{video_id}/annotations/{annotation_id}", status_code=204)
def delete_annotation(video_id: str, annotation_id: str, s: Session = Depends(db)):
    a = get_or_404(s, Annotation, annotation_id)
    if a.video_id != video_id:
        raise HTTPException(404, "Annotation not found for this video")
    s.delete(a)
    s.commit()
