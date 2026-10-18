"""FastAPI application entry point.

Run (from the repository root):
    .venv\\Scripts\\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8100
OpenAPI docs: http://127.0.0.1:8100/docs
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import analysis, capture, decisions, jobs, labeling, matches, multiview, proposals, system, videos
from backend.app.api.deps import auth_role
from backend.app.core.config import get_settings
from backend.app.database.models import init_db
from backend.app.workers.jobs import JobManager


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k in ("request_id", "path", "status", "duration_ms"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        return json.dumps(payload)


def _setup_logging(level: str) -> None:
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    root = logging.getLogger("drs")
    root.handlers[:] = [h]
    root.setLevel(level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    _setup_logging(s.log_level)
    init_db(s.db_url)
    app.state.jobs = JobManager(s)
    n = app.state.jobs.recover()
    if n:
        logging.getLogger("drs").warning("marked %d stale job(s) as interrupted", n)
    app.state.capture = None
    app.state.capture_cfg = {"root": s.capture_dir, "process_per_camera": s.capture_process_per_camera,
                             "recorder": {"retention_s": s.capture_retention_s, "max_disk_mb": s.capture_max_disk_mb,
                                          "encoder": s.capture_encoder}}
    yield
    if app.state.capture is not None:
        app.state.capture.stop_all()
    app.state.jobs.shutdown()


app = FastAPI(
    title="Cricket DRS research prototype API",
    version="0.1.0",
    description="Experimental ball tracking and decision-review analysis. Not an official decision system.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=list(get_settings().cors_origins), allow_methods=["*"],
                   allow_headers=["*"], expose_headers=["X-Request-ID", "X-Frame-Timestamp-Ms", "X-Timestamp-Source"])
access_log = logging.getLogger("drs.access")


@app.middleware("http")
async def request_id_mw(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        access_log.exception("unhandled error", extra={"request_id": rid, "path": request.url.path})
        response = JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)
    response.headers["X-Request-ID"] = rid
    access_log.info("%s %s", request.method, request.url.path,
                    extra={"request_id": rid, "path": request.url.path, "status": response.status_code,
                           "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
    return response


for r in (system.router, videos.router, jobs.router, analysis.router, matches.router, decisions.router, capture.router, labeling.router, proposals.router, multiview.router):
    app.include_router(r, dependencies=[Depends(auth_role)])
