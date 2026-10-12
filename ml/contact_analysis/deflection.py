"""Experimental bat-ball contact *candidate* analysis from ball trajectory changes.

Looks for frames where the ball's image-space velocity changes direction or
speed more than the local motion model predicts. A detected deflection is only
a candidate: bounces, pad contact, glove contact, tracker switches and
detection noise produce the same signature, and an absent deflection does not
rule out a fine edge. With low frame rates the result is INCONCLUSIVE.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

from ml.common.schemas import BallTrack

MIN_FPS_FOR_CONTACT = 100.0


class ContactCandidate(BaseModel):
    frame_index: int
    timestamp_ms: float
    direction_change_deg: float
    speed_ratio: float
    score: float
    relevant_frames: list[int]


class ContactResult(BaseModel):
    status: str  # CANDIDATES_FOUND | NO_DEFLECTION_DETECTED | INCONCLUSIVE | INSUFFICIENT_DATA
    candidates: list[ContactCandidate] = Field(default_factory=list)
    frame_rate_fps: float | None
    observations_used: int
    alternative_explanations: list[str] = Field(default_factory=lambda: [
        "Ball bounce on the pitch", "Contact with pad, body or glove", "Tracker switched to another object",
        "Detection noise or motion blur", "Camera motion",
    ])
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str = "A trajectory change does not prove bat contact; its absence does not prove no contact."


def analyze_deflection(track: BallTrack, fps: float, window: int = 6, min_angle_deg: float = 12.0) -> ContactResult:
    obs = [p for p in track.points if p.kind == "observed"]
    limitations = []
    if fps < MIN_FPS_FOR_CONTACT:
        limitations.append(
            f"Frame rate {fps:.0f} fps: the ball travels roughly {38 / max(fps, 1):.2f} m between frames at 140 km/h, "
            "so contact timing and small deflections cannot be resolved"
        )
    if len(obs) < 2 * window + 1:
        return ContactResult(status="INSUFFICIENT_DATA", frame_rate_fps=fps, observations_used=len(obs),
                             limitations=limitations + [f"Need at least {2 * window + 1} observed points"])
    t = np.array([p.timestamp_ms / 1000 for p in obs])
    xy = np.array([[p.x, p.y] for p in obs])
    cands: list[ContactCandidate] = []
    for i in range(window, len(obs) - window):
        before, after = slice(i - window, i + 1), slice(i, i + window + 1)
        vb = np.polyfit(t[before], xy[before], 1)[0]
        va = np.polyfit(t[after], xy[after], 1)[0]
        nb, na = np.linalg.norm(vb), np.linalg.norm(va)
        if nb < 1e-6 or na < 1e-6:
            continue
        ang = math.degrees(math.acos(np.clip(np.dot(vb, va) / (nb * na), -1, 1)))
        ratio = na / nb
        if ang >= min_angle_deg or not (0.6 <= ratio <= 1.6):
            score = ang / 90 + abs(math.log(ratio))
            cands.append(ContactCandidate(frame_index=obs[i].frame_index, timestamp_ms=obs[i].timestamp_ms,
                                          direction_change_deg=ang, speed_ratio=ratio, score=score,
                                          relevant_frames=[obs[k].frame_index for k in range(i - 2, i + 3)]))
    # Non-maximum suppression in time.
    cands.sort(key=lambda c: -c.score)
    kept: list[ContactCandidate] = []
    for c in cands:
        if all(abs(c.frame_index - k.frame_index) > window for k in kept):
            kept.append(c)
    if fps < MIN_FPS_FOR_CONTACT:
        status = "INCONCLUSIVE"
    else:
        status = "CANDIDATES_FOUND" if kept else "NO_DEFLECTION_DETECTED"
    return ContactResult(status=status, candidates=kept[:5], frame_rate_fps=fps, observations_used=len(obs),
                         limitations=limitations + ["No audio or bat detection is used", "Image-space analysis only"])
