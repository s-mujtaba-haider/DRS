"""Background job execution.

Jobs run in a bounded thread pool (default one worker so GPU memory stays
predictable). State is persisted in the database on every transition, so the
API can report progress and a restarted server marks orphaned jobs as
``interrupted`` instead of pretending they are still running.
"""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.database.models import Job, SessionLocal, Video

log = logging.getLogger("drs.jobs")

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool = ThreadPoolExecutor(max_workers=settings.worker_threads, thread_name_prefix="drs-job")
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def recover(self) -> int:
        with SessionLocal() as s:
            stale = s.query(Job).filter(Job.status.in_(["queued", "running"])).all()
            for j in stale:
                j.status = "interrupted"
                j.error = "Server restarted before the job finished; resubmit to rerun"
                j.finished_at = datetime.now(UTC)
            s.commit()
            return len(stale)

    def submit(self, job_id: str) -> None:
        with self._lock:
            self._cancel[job_id] = threading.Event()
        self.pool.submit(self._run, job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            ev = self._cancel.get(job_id)
        if ev is None:
            return False
        ev.set()
        with SessionLocal() as s:
            j = s.get(Job, job_id)
            if j and j.status == "queued":
                j.status, j.finished_at = "cancelled", datetime.now(UTC)
                s.commit()
        return True

    def _update(self, job_id: str, **kw) -> None:
        with SessionLocal() as s:
            j = s.get(Job, job_id)
            if j is None:
                return
            for k, v in kw.items():
                setattr(j, k, v)
            s.commit()

    def job_dir(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id

    def _run(self, job_id: str) -> None:
        from ml.pipeline import CancelledError, PipelineConfig, run_pipeline

        ev = self._cancel.get(job_id) or threading.Event()
        with SessionLocal() as s:
            job = s.get(Job, job_id)
            if job is None or job.status != "queued":
                return
            video = s.get(Video, job.video_id)
            cfg_dict = dict(job.config)
        if ev.is_set():
            self._update(job_id, status="cancelled", finished_at=datetime.now(UTC))
            return
        self._update(job_id, status="running", started_at=datetime.now(UTC), stage="starting")
        last = {"p": -1.0}

        def progress(stage: str, frac: float) -> None:
            if abs(frac - last["p"]) >= 0.02 or stage != last.get("s"):
                last.update(p=frac, s=stage)
                self._update(job_id, stage=stage, progress=float(frac))

        try:
            allowed = {f.name for f in fields(PipelineConfig)}
            cfg = PipelineConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})
            summary = run_pipeline(self.settings.videos_dir / video.stored_name, self.job_dir(job_id), cfg,
                                   progress=progress, should_cancel=ev.is_set)
            self._update(job_id, status="succeeded", progress=1.0, stage="done", summary=summary,
                         finished_at=datetime.now(UTC))
        except CancelledError:
            self._update(job_id, status="cancelled", finished_at=datetime.now(UTC))
        except Exception as exc:  # job failures are reported, never crash the server
            log.error("job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            self._update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}",
                         finished_at=datetime.now(UTC))
        finally:
            with self._lock:
                self._cancel.pop(job_id, None)

    def shutdown(self) -> None:
        for ev in list(self._cancel.values()):
            ev.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
