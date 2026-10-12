"""SQLAlchemy persistence (SQLite by default; any SQLAlchemy URL works, e.g. PostgreSQL)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _id() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_name: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    meta: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="ball_tracking")
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|succeeded|failed|cancelled|interrupted
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    config: Mapped[dict] = mapped_column(JSON)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Calibration(Base):
    __tablename__ = "calibrations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    video_id: Mapped[str | None] = mapped_column(ForeignKey("videos.id"), nullable=True, index=True)
    camera: Mapped[dict] = mapped_column(JSON)  # CameraModel
    correspondences: Mapped[dict] = mapped_column(JSON)  # inputs, preserved for reproducibility
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Annotation(Base):
    """Manual corrections. Never overwrite model output; stored alongside it."""

    __tablename__ = "annotations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    frame_index: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(32), default="ball")
    action: Mapped[str] = mapped_column(String(16))  # add|move|reject|mark_absent
    data: Mapped[dict] = mapped_column(JSON)
    author: Mapped[str] = mapped_column(String(64), default="local-user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class LbwAnalysis(Base):
    __tablename__ = "lbw_analyses"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    calibration_id: Mapped[str | None] = mapped_column(ForeignKey("calibrations.id"), nullable=True)
    request: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


_engine = None
SessionLocal = sessionmaker(expire_on_commit=False)


def init_db(url: str):
    global _engine
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    _engine = create_engine(url, **kwargs)
    Base.metadata.create_all(_engine)
    SessionLocal.configure(bind=_engine)
    return _engine
