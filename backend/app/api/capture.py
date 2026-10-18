"""Live multi-camera capture (rolling buffer, preview, instant clip export) and time synchronisation.

All cameras share one host clock. This is *software* synchronisation (frame arrival time), so every
exported clip carries its sync uncertainty, and multi-view reconstruction must treat it as degraded
unless hardware-synchronised timestamps are supplied.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.app.api.deps import db, get_or_404, settings, to_dict
from backend.app.core.config import Settings
from backend.app.database.models import Review, ReviewEvent, Video

router = APIRouter(prefix="/api", tags=["capture"])
_LOCK = threading.Lock()
_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_URL = re.compile(r"^(rtsp|rtsps|http|https)://", re.I)


def _manager(request: Request, create: bool = False):
    mgr = getattr(request.app.state, "capture", None)
    if mgr is None and create:
        from ml.capture.manager import CaptureManager
        from ml.capture.recorder import RecorderConfig

        cfg = request.app.state.capture_cfg
        mgr = CaptureManager(cfg["root"], RecorderConfig(**cfg["recorder"]), process_per_camera=cfg["process_per_camera"])
        request.app.state.capture = mgr
    return mgr


class CameraSpec(BaseModel):
    camera_id: str
    type: Literal["opencv", "synthetic", "video"] = "opencv"
    uri: int | str | None = Field(None, description="USB device index or rtsp/http(s) stream URL (no local file paths)")
    video_id: str | None = Field(None, description="type=video: replay an uploaded video as a live camera (rehearsal)")
    loop: bool = True
    width: int | None = Field(None, ge=64, le=7680)
    height: int | None = Field(None, ge=64, le=4320)
    fps: float | None = Field(None, gt=0, le=1000)

    @field_validator("camera_id")
    @classmethod
    def _cid(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError("camera_id: 1-32 letters, digits, '_' or '-'")
        return v

    @field_validator("uri")
    @classmethod
    def _uri(cls, v):
        if isinstance(v, str):
            if v.isdigit():
                return int(v)
            if not _URL.match(v):
                raise ValueError("uri must be a device index or an rtsp/http(s) URL")
        return v


class AudioSpec(BaseModel):
    type: Literal["device", "synthetic"] = "device"
    device: int | None = Field(None, description="PortAudio input index (GET /api/capture/audio-devices); None = default")
    sample_rate: int = Field(48000, ge=8000, le=192000)


class CaptureStart(BaseModel):
    cameras: list[CameraSpec] = Field(default_factory=list, max_length=8)
    audio: AudioSpec | None = None
    retention_s: float | None = Field(None, ge=10, le=3600)


@router.get("/capture/audio-devices")
def audio_devices():
    from ml.capture.audio import list_input_devices

    return list_input_devices()


def _spec(c: CameraSpec, s: Session, cfg: Settings) -> dict:
    if c.type == "opencv":
        if c.uri is None:
            raise HTTPException(422, f"{c.camera_id}: opencv cameras need a uri")
        return c.model_dump(exclude_none=True, exclude={"video_id", "loop"})
    if c.type == "video":
        if not c.video_id:
            raise HTTPException(422, f"{c.camera_id}: video cameras need a video_id")
        v = get_or_404(s, Video, c.video_id)
        path = (cfg.videos_dir / v.stored_name).resolve()
        if path.parent != cfg.videos_dir.resolve() or not path.is_file():
            raise HTTPException(404, "Video file missing")
        # Server-resolved path of an uploaded video only; clients never supply file paths.
        return {"camera_id": c.camera_id, "type": "opencv", "uri": str(path), "loop": c.loop, "realtime_file": True}
    return c.model_dump(exclude_none=True, exclude={"video_id", "loop", "uri"})


@router.post("/capture/start")
def capture_start(body: CaptureStart, request: Request, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    if len({c.camera_id for c in body.cameras}) != len(body.cameras):
        raise HTTPException(422, "camera_id values must be unique")
    if not body.cameras and body.audio is None:
        raise HTTPException(422, "Give at least one camera or an audio source")
    specs = [_spec(c, s, cfg) for c in body.cameras]
    with _LOCK:
        mgr = _manager(request, create=True)
        running = {cid for cid, r in mgr.recorders.items() if r.is_alive()}
        clash = running & {c.camera_id for c in body.cameras}
        if clash:
            raise HTTPException(409, f"Already recording: {', '.join(sorted(clash))}")
        rec_cfg = None
        if body.retention_s:
            from dataclasses import replace

            rec_cfg = replace(mgr.cfg, retention_s=body.retention_s)
        try:
            if specs:
                mgr.start_cameras(specs, rec_cfg)
            if body.audio is not None:
                from ml.capture.audio import SoundDeviceSource, SyntheticAudioSource

                src = (SyntheticAudioSource(sample_rate=body.audio.sample_rate) if body.audio.type == "synthetic"
                       else SoundDeviceSource(body.audio.device, body.audio.sample_rate))
                mgr.start_audio(src)
        except HTTPException:
            raise
        except Exception as exc:  # device open / encoder failures are user-facing
            raise HTTPException(422, f"Could not start capture: {type(exc).__name__}: {exc}") from exc
    return mgr.status()


@router.post("/capture/stop")
def capture_stop(request: Request, camera_id: str | None = None):
    mgr = _manager(request)
    if mgr is None:
        return {"cameras": []}
    with _LOCK:
        if camera_id:
            if camera_id not in mgr.recorders:
                raise HTTPException(404, f"camera {camera_id} not found")
            mgr.stop_camera(camera_id)
        else:
            mgr.stop_all()
    return mgr.status()


@router.get("/capture/status")
def capture_status(request: Request):
    mgr = _manager(request)
    if mgr is None:
        return {"sync_method": "software_shared_clock", "hardware_sync": False, "cameras": []}
    return mgr.status()


@router.get("/capture/preview/{camera_id}")
def capture_preview(camera_id: str, request: Request):
    mgr = _manager(request)
    if mgr is None:
        raise HTTPException(404, "capture not started")
    try:
        jpg = mgr.latest_frame(camera_id)
    except KeyError as exc:
        raise HTTPException(404, f"camera {camera_id} not found") from exc
    if jpg is None:
        raise HTTPException(404, "no frame yet")
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


class ClipRequest(BaseModel):
    seconds: float = Field(8.0, gt=0.2, le=120)
    camera_ids: list[str] | None = None
    review_id: str | None = None
    label: str | None = Field(None, max_length=80)


def _import_clip(s: Session, cfg: Settings, cam_id: str, entry: dict, group: str, label: str | None,
                 audio: dict | None = None) -> Video:
    from ml.video.ingest import probe_video

    src = Path(entry["clip_path"])
    stored = f"{uuid.uuid4().hex}.mp4"
    dest = cfg.videos_dir / stored
    shutil.move(str(src), dest)
    h = hashlib.sha256()
    with dest.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    timing = Path(entry["timing_path"])
    timing_dest = cfg.videos_dir / f"{Path(stored).stem}.timing.json"
    shutil.move(str(timing), timing_dest)
    meta = probe_video(dest, full_scan=True)  # clips are short; an exact frame count matters for review
    name = f"{label or 'capture'}_{cam_id}.mp4"
    meta.path_name = name
    t = json.loads(timing_dest.read_text())
    m = meta.model_dump(mode="json")
    m["capture"] = {"camera_id": cam_id, "clip_group": group, "time_offset_ms": entry.get("time_offset_ms"),
                    "sync_uncertainty_ms": entry.get("sync_uncertainty_ms"), "hardware_sync": False,
                    "gaps": len(entry.get("gaps") or []), "missing_frames": entry.get("missing_frames"),
                    "timing_file": timing_dest.name, "first_host_ns": entry.get("first_host_ns"),
                    "clip_start_utc_ns": (t.get("window") or {}).get("t_start_utc_ns")}
    if audio and audio.get("status") == "ok":
        # Audio sample 0 on this clip's time axis: both are offsets from the same export window start.
        m["capture"]["audio"] = {"file": audio["file"], "offset_ms": round(audio["offset_ms"] - (entry.get("time_offset_ms") or 0.0), 3),
                                 "sample_rate": audio["sample_rate"], "timestamp_basis": audio.get("timestamp_basis"),
                                 "uncertainty_ms": round(audio["uncertainty_ms"] + (entry.get("sync_uncertainty_ms") or 0.0), 3)}
    if entry.get("missing_frames"):
        m["warnings"] = [*m.get("warnings", []), f"{entry['missing_frames']} frame(s) missing in capture (dropped)"]
    v = Video(original_filename=name, stored_name=stored, size_bytes=dest.stat().st_size, sha256=h.hexdigest(), meta=m)
    s.add(v)
    return v


@router.post("/capture/clip", status_code=201)
def capture_clip(body: ClipRequest, request: Request, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    """Export the last N seconds from every (or the chosen) camera as videos, optionally attached to a review."""
    mgr = _manager(request)
    if mgr is None or not mgr.recorders:
        raise HTTPException(409, "No cameras are recording")
    review = None
    if body.review_id:
        review = get_or_404(s, Review, body.review_id)
        if review.status != "open":
            raise HTTPException(409, f"Review is {review.status}")
    group = uuid.uuid4().hex[:12]
    tmp = cfg.storage_dir / "clips_tmp" / group
    try:
        exp = mgr.export_last(body.seconds, tmp, body.camera_ids)
    except KeyError as exc:
        raise HTTPException(404, f"camera {exc} not found") from exc
    audio = exp.get("audio")
    if audio and audio.get("status") == "ok":
        dest = cfg.videos_dir / f"{group}.audio.wav"
        shutil.move(audio["path"], dest)
        audio = {**audio, "file": dest.name, "timestamp_basis": (mgr.audio.status() if mgr.audio else {}).get("timestamp_basis")}
    videos, failed = [], {}
    for cam_id, entry in exp["cameras"].items():
        if entry.get("status") != "ok" or not entry.get("clip_path"):
            failed[cam_id] = entry.get("status")
            continue
        videos.append(_import_clip(s, cfg, cam_id, entry, group, body.label, audio))
    s.flush()
    if review is not None and videos:
        review.video_ids = [*review.video_ids, *[v.id for v in videos]]
        s.add(ReviewEvent(review_id=review.id, action="capture_clip_added", actor="operator",
                          payload={"clip_group": group, "video_ids": [v.id for v in videos], "seconds": body.seconds}))
    s.commit()
    shutil.rmtree(tmp, ignore_errors=True)
    if not videos:
        raise HTTPException(422, f"No clip could be exported: {failed}")
    return {"clip_group": group, "videos": [to_dict(v) for v in videos], "failed": failed,
            "sync": exp.get("sync"), "audio": {k: v for k, v in (audio or {}).items() if k != "path"} or None,
            "warnings": exp.get("warnings", [])}


# ---------------------------------------------------------------- synchronisation
class AudioSyncRequest(BaseModel):
    reference_video_id: str
    other_video_id: str
    max_offset_ms: float = Field(2000, gt=0, le=30000)


@router.post("/sync/audio")
def sync_audio(body: AudioSyncRequest, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    """Offset of the other video relative to the reference from audio (GCC-PHAT): t_ref = t_other + offset."""
    from ml.synchronization import audio_offset

    a, b = get_or_404(s, Video, body.reference_video_id), get_or_404(s, Video, body.other_video_id)
    res = audio_offset(cfg.videos_dir / a.stored_name, cfg.videos_dir / b.stored_name, max_offset_ms=body.max_offset_ms)
    return res.model_dump(mode="json")


class EventSyncRequest(BaseModel):
    reference_video_id: str
    other_video_id: str
    events_reference_ms: list[float | tuple[float, float]] = Field(min_length=1)
    events_other_ms: list[float | tuple[float, float]] = Field(min_length=1)


@router.post("/sync/events")
def sync_events(body: EventSyncRequest, s: Session = Depends(db)):
    """Offset from matched visual events (e.g. bail break, bounce) marked in both videos."""
    from ml.synchronization import event_offset

    if len(body.events_reference_ms) != len(body.events_other_ms):
        raise HTTPException(422, "events must be matched one-to-one")
    a, b = get_or_404(s, Video, body.reference_video_id), get_or_404(s, Video, body.other_video_id)
    fa, fb = a.meta.get("nominal_fps"), b.meta.get("nominal_fps")
    if not (fa and fb):
        raise HTTPException(422, "both videos need a frame rate")
    return event_offset(body.events_reference_ms, body.events_other_ms, fa, fb).model_dump(mode="json")
