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


MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def auth_role(request: Request) -> str:
    """Match-day roles from the X-DRS-PIN header. Open (no PINs configured) means single-user local mode."""
    import hmac

    s = get_settings()
    enabled = bool(s.operator_pin or s.umpire_pin)
    pin = request.headers.get("X-DRS-PIN") or ""
    role = "open"
    if enabled:
        role = None
        if pin and s.umpire_pin and hmac.compare_digest(pin, s.umpire_pin):
            role = "umpire"
        elif pin and s.operator_pin and hmac.compare_digest(pin, s.operator_pin):
            role = "operator"
    request.state.auth_enabled, request.state.role = enabled, role
    if role is None and request.method in MUTATING:
        raise HTTPException(401, "A valid operator or umpire PIN is required (header X-DRS-PIN)")
    return role or "viewer"


def require_umpire(request: Request) -> None:
    if getattr(request.state, "auth_enabled", False) and request.state.role != "umpire":
        raise HTTPException(403, "Only the third umpire (umpire PIN) can record or withdraw a decision")
