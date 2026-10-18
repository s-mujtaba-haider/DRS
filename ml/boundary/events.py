"""Observed event sequence near the boundary (evidence layer, no Law logic).

Each event is something an operator (or a future detector) observed on the frames, with its
time and the ground-contact positions that matter. Positions are measured against the rope by
``ml.boundary.geometry``; the Law 19 / Law 33 interpretation lives in ``ml.boundary.rules``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class GroundPoint(BaseModel):
    point: tuple[float, float]
    frame: Literal["image", "world"] = "image"
    sigma: float | None = Field(None, gt=0, description="localisation sigma (px or m); default from config")


class _Event(BaseModel):
    timestamp_ms: float
    frame_index: int | None = Field(None, ge=0)
    source: Literal["manual", "model"] = "manual"
    note: str | None = None


class BallGroundContact(_Event):
    kind: Literal["ball_ground"] = "ball_ground"
    position: GroundPoint | None = None


class FielderBallContact(_Event):
    kind: Literal["fielder_ball"] = "fielder_ball"
    fielder_id: str
    airborne: bool | None = Field(None, description="fielder not in contact with the ground at the touch")
    ground_contacts: list[GroundPoint] = Field(default_factory=list, description="body parts on the ground at the touch (not airborne)")
    last_ground_contact: GroundPoint | None = Field(None, description="airborne: final ground contact before this touch")
    touching_rope: bool | None = Field(None, description="fielder in contact with the rope/marker or an object beyond it")


class FielderGroundContact(_Event):
    kind: Literal["fielder_ground"] = "fielder_ground"
    fielder_id: str
    position: GroundPoint | None = None
    in_contact_with_ball: bool | None = None


class FielderRopeContact(_Event):
    kind: Literal["fielder_rope"] = "fielder_rope"
    fielder_id: str
    in_contact_with_ball: bool | None = None


class BallCrossedRope(_Event):
    kind: Literal["ball_crossed_rope"] = "ball_crossed_rope"
    in_air: bool | None = None


class CatchCompleted(_Event):
    kind: Literal["catch_completed"] = "catch_completed"
    fielder_id: str


class BallDead(_Event):
    kind: Literal["ball_dead"] = "ball_dead"


BoundaryEvent = Annotated[
    BallGroundContact | FielderBallContact | FielderGroundContact | FielderRopeContact | BallCrossedRope | CatchCompleted | BallDead,
    Field(discriminator="kind"),
]
