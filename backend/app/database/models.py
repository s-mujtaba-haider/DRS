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


class DecisionAnalysis(Base):
    """Evidence analyses other than LBW (run-out, stumping, boundary, catch, wicket break, sync)."""

    __tablename__ = "decision_analyses"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    video_id: Mapped[str | None] = mapped_column(ForeignKey("videos.id"), nullable=True, index=True)
    calibration_id: Mapped[str | None] = mapped_column(ForeignKey("calibrations.id"), nullable=True)
    request: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    name: Mapped[str] = mapped_column(String(200))
    team_a: Mapped[str] = mapped_column(String(100))
    team_b: Mapped[str] = mapped_column(String(100))
    protocol_id: Mapped[str] = mapped_column(String(64))
    venue: Mapped[str | None] = mapped_column(String(200), nullable=True)
    match_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="live")  # live|completed|abandoned
    current_innings: Mapped[int] = mapped_column(Integer, default=1)
    batting_team: Mapped[str | None] = mapped_column(String(100), nullable=True)
    officials: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Review(Base):
    """One decision review. The final decision is always entered by a human official."""

    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    innings: Mapped[int] = mapped_column(Integer)
    over_ball: Mapped[str | None] = mapped_column(String(16), nullable=True)
    requested_by: Mapped[str] = mapped_column(String(100))  # team name, or "umpire"
    kind: Mapped[str] = mapped_column(String(16))  # player|umpire
    decision_type: Mapped[str] = mapped_column(String(32))  # lbw|caught|run_out|stumping|boundary|other
    on_field: Mapped[str] = mapped_column(String(16))  # OUT|NOT_OUT|NONE (umpire referral without soft signal)
    batter: Mapped[str | None] = mapped_column(String(100), nullable=True)
    bowler: Mapped[str | None] = mapped_column(String(100), nullable=True)
    video_ids: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[list] = mapped_column(JSON, default=list)  # [{type, id, attached_at}]
    recommendation: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # system output, advisory only
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|decided|withdrawn
    final_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)  # OUT|NOT_OUT (human)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)  # UPHELD|OVERTURNED|UMPIRES_CALL|INCONCLUSIVE
    decided_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ReviewEvent(Base):
    """Append-only audit log for a review (who did what, when, with which payload)."""

    __tablename__ = "review_events"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    review_id: Mapped[str] = mapped_column(ForeignKey("reviews.id"), index=True)
    action: Mapped[str] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(100), default="operator")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
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
