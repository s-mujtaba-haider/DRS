"""Shared data schemas for the DRS pipeline.

These pydantic models are the contract between ML modules, the backend API
and exported files. Every point carries an explicit `kind` so observations,
filtered estimates, interpolations, predictions and manual corrections are
never confused.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TimestampSource(StrEnum):
    CONTAINER_PTS = "container_pts"  # presentation timestamp read from the container
    ESTIMATED_FROM_FPS = "estimated_from_fps"  # index / nominal fps; not ground truth


class VideoMetadata(BaseModel):
    path_name: str
    container_format: str | None = None
    codec: str | None = None
    width: int
    height: int
    nominal_fps: float | None = Field(None, description="Declared average frame rate (may be unreliable)")
    frame_count_declared: int | None = None
    frame_count_decoded: int | None = None
    duration_s: float | None = None
    time_base: str | None = None
    variable_frame_rate: bool | None = Field(None, description="True if sampled PTS deltas vary materially")
    pts_delta_ms_mean: float | None = None
    pts_delta_ms_std: float | None = None
    has_audio: bool = False
    size_bytes: int | None = None
    warnings: list[str] = Field(default_factory=list)


class Detection(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_ms: float
    timestamp_source: TimestampSource = TimestampSource.CONTAINER_PTS
    bbox: tuple[float, float, float, float] = Field(description="x1, y1, x2, y2 in pixels")
    center: tuple[float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    model_id: str
    source: str = Field("model", description="model | motion | manual")
    label: str = "ball"
    uncertainty_px: float | None = None

    @field_validator("bbox")
    @classmethod
    def _valid_bbox(cls, v: tuple[float, float, float, float]):
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
        return v

    @model_validator(mode="after")
    def _center_inside(self):
        x1, y1, x2, y2 = self.bbox
        cx, cy = self.center
        if not (x1 - 1e-6 <= cx <= x2 + 1e-6 and y1 - 1e-6 <= cy <= y2 + 1e-6):
            raise ValueError("center must lie inside bbox")
        return self


class FrameDetections(BaseModel):
    frame_index: int
    timestamp_ms: float
    timestamp_source: TimestampSource
    ball_detected: bool
    detections: list[Detection] = Field(default_factory=list)


PointKind = Literal["observed", "filtered", "interpolated", "predicted", "manual"]


class TrackPoint(BaseModel):
    frame_index: int
    timestamp_ms: float
    kind: PointKind
    x: float
    y: float
    sigma_x: float | None = None
    sigma_y: float | None = None
    confidence: float | None = None
    detection_source: str | None = None


class TrackQuality(BaseModel):
    num_frames_spanned: int
    num_observed: int
    num_missing: int
    observed_ratio: float
    longest_gap: int
    mean_innovation_px: float | None
    mean_confidence: float | None
    fragments_merged: int = 0
    reliable: bool = Field(False, description="Passes minimum consistency checks; NOT a correctness guarantee")
    reliability_issues: list[str] = Field(default_factory=list)


class BallTrack(BaseModel):
    track_id: int
    is_primary: bool = False
    points: list[TrackPoint]
    quality: TrackQuality
    score: float


class TrackingResult(BaseModel):
    tracks: list[BallTrack]
    primary_track_id: int | None
    num_frames: int
    num_frames_with_candidates: int
    num_tracks_total: int
    fragmentation_note: str | None = None
    warnings: list[str] = Field(default_factory=list)
