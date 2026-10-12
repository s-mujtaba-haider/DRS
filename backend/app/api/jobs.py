"""Analysis jobs: create, status, cancel, results, artifacts, exports, PDF report."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.api.deps import db, get_or_404, jobs, settings, to_dict
from backend.app.core.config import Settings
from backend.app.database.models import Job, LbwAnalysis, Video
from backend.app.workers.jobs import TERMINAL, JobManager

router = APIRouter(prefix="/api/analysis/jobs", tags=["jobs"])

ARTIFACTS = {
    "annotated.mp4": "video/mp4",
    "summary.json": "application/json",
    "tracking.json": "application/json",
    "detections.json": "application/json",
    "trajectory_2d.json": "application/json",
    "metadata.json": "application/json",
    "track_points.csv": "text/csv",
}


class JobCreate(BaseModel):
    video_id: str
    detector: str = Field("auto", pattern="^(auto|cricket|coco_hybrid|coco|motion)$")
    conf: float = Field(0.15, ge=0.01, le=0.99)
    imgsz: int = Field(1280, ge=320, le=3840)
    tile_size: int | None = Field(None, ge=320, le=2048)
    batch_size: int = Field(8, ge=1, le=64)
    device: str = Field("auto", pattern=r"^(auto|cpu|\d)$")
    half: bool = True
    frame_step: int = Field(1, ge=1, le=30)
    start_frame: int = Field(0, ge=0)
    end_frame: int | None = Field(None, ge=1)
    max_frames: int | None = Field(None, ge=1)
    render_video: bool = True


@router.post("", status_code=201)
def create_job(body: JobCreate, s: Session = Depends(db), mgr: JobManager = Depends(jobs), cfg: Settings = Depends(settings)):
    get_or_404(s, Video, body.video_id)
    if body.end_frame is not None and body.end_frame <= body.start_frame:
        raise HTTPException(422, "end_frame must be greater than start_frame")
    conf = body.model_dump(exclude={"video_id"})
    conf["max_frames"] = min(conf["max_frames"] or cfg.max_frames_per_job, cfg.max_frames_per_job)
    j = Job(video_id=body.video_id, config=conf)
    s.add(j)
    s.commit()
    mgr.submit(j.id)
    return to_dict(j)


@router.get("")
def list_jobs(video_id: str | None = None, s: Session = Depends(db)):
    q = s.query(Job)
    if video_id:
        q = q.filter(Job.video_id == video_id)
    return [to_dict(j) for j in q.order_by(Job.created_at.desc()).limit(200).all()]


@router.get("/{job_id}")
def get_job(job_id: str, s: Session = Depends(db)):
    return to_dict(get_or_404(s, Job, job_id))


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, s: Session = Depends(db), mgr: JobManager = Depends(jobs)):
    j = get_or_404(s, Job, job_id)
    if j.status in TERMINAL:
        raise HTTPException(409, f"Job already {j.status}")
    mgr.cancel(job_id)
    s.refresh(j)
    return to_dict(j)


def _job_dir_ok(job_id: str, s: Session, cfg: Settings):
    j = get_or_404(s, Job, job_id)
    if j.status != "succeeded":
        raise HTTPException(409, f"Job is {j.status}; results are available only for succeeded jobs")
    return j, cfg.jobs_dir / job_id


@router.get("/{job_id}/results")
def job_results(job_id: str, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    j, d = _job_dir_ok(job_id, s, cfg)
    out = {"job": to_dict(j), "summary": j.summary}
    for name in ("tracking.json", "trajectory_2d.json", "metadata.json"):
        p = d / name
        out[name.removesuffix(".json")] = json.loads(p.read_text()) if p.is_file() else None
    return out


@router.get("/{job_id}/detections")
def job_detections(job_id: str, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    _, d = _job_dir_ok(job_id, s, cfg)
    return json.loads((d / "detections.json").read_text())


@router.get("/{job_id}/artifacts/{name}")
def job_artifact(job_id: str, name: str, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    if name not in ARTIFACTS:
        raise HTTPException(404, "Unknown artifact")
    _, d = _job_dir_ok(job_id, s, cfg)
    p = d / name
    if not p.is_file():
        raise HTTPException(404, f"{name} was not produced by this job")
    return FileResponse(p, media_type=ARTIFACTS[name], filename=f"{job_id}_{name}")


@router.get("/{job_id}/report.pdf")
def job_report(job_id: str, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    import cv2

    from backend.app.services.reports import build_job_report
    from ml.video.ingest import iter_frames

    j, d = _job_dir_ok(job_id, s, cfg)
    v = s.get(Video, j.video_id)
    lbw = s.query(LbwAnalysis).filter(LbwAnalysis.job_id == job_id).order_by(LbwAnalysis.created_at.desc()).first()
    frame_png = None
    tracking = json.loads((d / "tracking.json").read_text())
    primary = next((t for t in tracking["tracks"] if t["is_primary"]), None)
    if primary and (d / "annotated.mp4").is_file():
        obs = [p["frame_index"] for p in primary["points"] if p["kind"] == "observed"]
        mid = obs[len(obs) // 2] - (j.config.get("start_frame") or 0)
        step = j.config.get("frame_step") or 1
        rec = next(iter_frames(d / "annotated.mp4", start_frame=max(mid // step, 0), max_frames=1), None)
        if rec is not None:
            frame_png = cv2.imencode(".png", rec.image)[1].tobytes()
    pdf = build_job_report(to_dict(j), to_dict(v), d, to_dict(lbw) if lbw else None, frame_png)
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="drs_report_{job_id}.pdf"'})


@router.get("/{job_id}/export.json")
def job_export(job_id: str, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    j, d = _job_dir_ok(job_id, s, cfg)
    lbw = s.query(LbwAnalysis).filter(LbwAnalysis.job_id == job_id).all()
    payload = {
        "job": to_dict(j),
        "tracking": json.loads((d / "tracking.json").read_text()),
        "trajectory_2d": json.loads((d / "trajectory_2d.json").read_text()) if (d / "trajectory_2d.json").is_file() else None,
        "lbw_analyses": [to_dict(a) for a in lbw],
    }
    return Response(json.dumps(payload, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="drs_{job_id}.json"'})
