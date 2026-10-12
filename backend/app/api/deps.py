from __future__ import annotations

from collections.abc import Iterator

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.database.models import SessionLocal
from backend.app.workers.jobs import JobManager


def db() -> Iterator[Session]:
    with SessionLocal() as s:
        yield s


def settings() -> Settings:
    return get_settings()


def jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def get_or_404(session: Session, model, obj_id: str):
    obj = session.get(model, obj_id)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} {obj_id} not found")
    return obj


def to_dict(obj) -> dict:
    out = {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
    for k, v in out.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out
