"""Crease-status engine at the moment the wicket is broken (run-out and stumping).

Answers one question: *was the batter in their ground when the wicket was put down?*
It combines a wicket-break interval (automatic candidate or operator-marked frames)
with per-frame batter observations (bat tip / foot ground-contact point and whether it
is grounded) and the crease geometry from ``ml.runout.crease``.

Laws used (MCC Laws of Cricket, 2017 Code 4th Edition, 2026,
https://www.lords.org/mcc/the-laws-of-cricket):
* Law 30.1.1: out of ground unless some part of person or bat is grounded behind the popping crease.
* Law 30.1.2: not out of ground if, running or diving towards that ground and beyond, having
  grounded beyond the crease, contact is subsequently lost ("bouncing bat").
* Law 29.1: the wicket is broken when a bail is completely removed or a stump removed.

Frame quantisation: video only shows the state at frame times. The break happened somewhere
in ``(last_intact_ms, first_broken_ms]``; the grounding happened somewhere between the last
frame not in ground and the first frame in ground. When these two windows overlap, the
order of events is not observable and the result is ``INCONCLUSIVE``, with both window
widths stated. At 25-50 fps the inter-frame gap is 20-40 ms.

Statuses:
* ``IN_GROUND``: grounded behind the crease in a frame no later than the last intact frame.
* ``OUT_OF_GROUND``: never in ground before the break, and observed definitely not behind the
  crease at or after the first broken frame.
* ``INCONCLUSIVE``: windows overlap, near-line spatial ambiguity, unknown grounding, or a
  Law 30.1.2 fact that must be supplied.
* ``INSUFFICIENT_DATA``: no break interval or no usable observations.

Nothing here is an umpire's decision.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ml.runout.crease import CreaseLine, CreaseMeasurement, measure_point

ENGINE_VERSION = "crease-engine-0.1.0-experimental"
DISCLAIMER = ("Experimental research output, not an official umpiring decision. Real-world accuracy of this "
              "system has not been established.")


class CreaseStatus(StrEnum):
    OUT_OF_GROUND = "OUT_OF_GROUND"
    IN_GROUND = "IN_GROUND"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


ObsState = Literal["IN", "NOT_IN_BEYOND", "AIRBORNE_BEHIND", "UNKNOWN"]


class WicketBreak(BaseModel):
    """The wicket was intact at ``last_intact`` and broken at ``first_broken``."""

    last_intact_frame: int = Field(ge=0)
    last_intact_ms: float
    first_broken_frame: int = Field(ge=0)
    first_broken_ms: float
    source: Literal["manual", "auto"] = "manual"
    confirmed: bool = Field(False, description="operator confirmed a bail completely removed / stump out (Law 29.1)")

    @model_validator(mode="after")
    def _order(self):
        if self.first_broken_frame <= self.last_intact_frame or self.first_broken_ms <= self.last_intact_ms:
            raise ValueError("first_broken must come after last_intact")
        return self

    @classmethod
    def from_detection(cls, det) -> WicketBreak | None:
        if det.status != "BREAK_CANDIDATE":
            return None
        return cls(last_intact_frame=det.last_intact_frame, last_intact_ms=det.last_intact_ms,
                   first_broken_frame=det.first_broken_frame, first_broken_ms=det.first_broken_ms, source="auto")


class BatterObservation(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_ms: float
    point: tuple[float, float] | None = Field(None, description="image xy of the relevant bat-tip / foot contact point")
    point_sigma_px: float | None = Field(None, gt=0)
    grounded: bool | None = Field(None, description="is that point in contact with the ground")
    source: Literal["manual", "model"] = "manual"
    note: str | None = None


class CreaseFacts(BaseModel):
    """Facts the video geometry cannot establish; None = unknown."""

    running_or_diving_into_ground: bool | None = Field(
        None, description="Law 30.1.2: batter was running/diving towards this ground and beyond")


class CreaseConfig(BaseModel):
    decision_confidence: float = Field(0.95, ge=0.5, lt=1.0)
    default_point_sigma_px: float = Field(2.0, gt=0)
    max_frame_gap_warn_ms: float = Field(10.0, gt=0)
    max_break_interval_warn_ms: float = Field(20.0, gt=0)


class ObservationAssessment(BaseModel):
    frame_index: int
    timestamp_ms: float
    state: ObsState
    grounded: bool | None
    source: str
    phase: Literal["before_break", "break_window", "after_break"]
    measurement: CreaseMeasurement | None = None
    note: str | None = None


class TimelineEvent(BaseModel):
    t_start_ms: float
    t_end_ms: float
    kind: str
    description: str
    frames: list[int] = Field(default_factory=list)
    source: str = "engine"


class CreaseEventResult(BaseModel):
    engine_version: str = ENGINE_VERSION
    status: CreaseStatus
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    assessments: list[ObservationAssessment] = Field(default_factory=list)
    break_window_ms: tuple[float, float] | None = None
    grounding_window_ms: tuple[float | None, float] | None = Field(
        None, description="(last not-in-ground, first in-ground) frame times; first is None if never observed out of ground")
    median_frame_gap_ms: float | None = None
    crease_source: str | None = None
    crease_end: str | None = None
    disclaimer: str = DISCLAIMER


def _assess(o: BatterObservation, line: CreaseLine, cfg: CreaseConfig) -> tuple[ObsState, CreaseMeasurement | None]:
    if o.point is None:
        return "UNKNOWN", None
    m = measure_point(line, o.point, o.point_sigma_px or cfg.default_point_sigma_px, cfg.decision_confidence)
    if m.state == "NOT_BEHIND":
        return "NOT_IN_BEYOND", m  # not behind the line: out of ground whether grounded or not
    if m.state == "AMBIGUOUS":
        return "UNKNOWN", m
    if o.grounded is True:
        return "IN", m
    if o.grounded is False:
        return "AIRBORNE_BEHIND", m
    return "UNKNOWN", m


def analyze_crease_event(
    line: CreaseLine,
    observations: list[BatterObservation],
    wicket_break: WicketBreak | None,
    facts: CreaseFacts | None = None,
    cfg: CreaseConfig | None = None,
) -> CreaseEventResult:
    facts, cfg = facts or CreaseFacts(), cfg or CreaseConfig()
    res = CreaseEventResult(status=CreaseStatus.INSUFFICIENT_DATA, crease_source=line.source, crease_end=line.end,
                            warnings=list(line.warnings))
    obs = sorted(observations, key=lambda o: o.timestamp_ms)
    times = np.array([o.timestamp_ms for o in obs])
    if len(times) >= 2:
        gaps = np.diff(times)
        res.median_frame_gap_ms = float(np.median(gaps))
        if res.median_frame_gap_ms > cfg.max_frame_gap_warn_ms:
            res.warnings.append(f"Observations are {res.median_frame_gap_ms:.1f} ms apart (~{1000 / res.median_frame_gap_ms:.0f} fps); "
                                "events between frames cannot be ordered more finely than this")
    if wicket_break is None:
        res.reasons.append("No wicket-break interval: mark the last intact and first broken frames, or run detection")
        return res
    t0, t1 = wicket_break.last_intact_ms, wicket_break.first_broken_ms
    res.break_window_ms = (t0, t1)
    res.timeline.append(TimelineEvent(t_start_ms=t0, t_end_ms=t1, kind="wicket_broken",
                                      description=f"Wicket broken within a {t1 - t0:.1f} ms window ({wicket_break.source}"
                                                  f"{', operator-confirmed' if wicket_break.confirmed else ', unconfirmed'})",
                                      frames=[wicket_break.last_intact_frame, wicket_break.first_broken_frame],
                                      source=wicket_break.source))
    if t1 - t0 > cfg.max_break_interval_warn_ms:
        res.warnings.append(f"Break window is {t1 - t0:.1f} ms wide; higher frame rate footage is needed to time it finely")
    if wicket_break.source == "auto" and not wicket_break.confirmed:
        res.warnings.append("Wicket break is an automatic candidate that has not been confirmed by the operator")

    for o in obs:
        state, m = _assess(o, line, cfg)
        phase = "before_break" if o.timestamp_ms <= t0 else ("after_break" if o.timestamp_ms >= t1 else "break_window")
        res.assessments.append(ObservationAssessment(frame_index=o.frame_index, timestamp_ms=o.timestamp_ms, state=state,
                                                     grounded=o.grounded, source=o.source, phase=phase, measurement=m, note=o.note))
    A = res.assessments
    if not any(a.state != "UNKNOWN" for a in A):
        res.reasons.append("No usable batter observation (point missing, grounding unknown, or on the line)")
        return res
    if any(a.source == "model" for a in A):
        res.warnings.append("Some observations come from an unvalidated model; check them on the frames")

    def first(pred, seq):
        return next((a for a in seq if pred(a)), None)

    first_in = first(lambda a: a.state == "IN", A)
    if first_in is not None:
        not_in_before = [a for a in A if a.timestamp_ms < first_in.timestamp_ms and a.state in ("NOT_IN_BEYOND", "AIRBORNE_BEHIND")]
        lo = not_in_before[-1].timestamp_ms if not_in_before else None
        res.grounding_window_ms = (lo, first_in.timestamp_ms)  # lo None: grounded in the first observation
        res.timeline.append(TimelineEvent(t_start_ms=lo if lo is not None else first_in.timestamp_ms, t_end_ms=first_in.timestamp_ms,
                                          kind="grounded_behind_crease", description="Bat or person first observed grounded behind the popping crease",
                                          frames=([not_in_before[-1].frame_index] if not_in_before else []) + [first_in.frame_index],
                                          source=first_in.source))
    res.timeline.sort(key=lambda e: (e.t_end_ms, e.t_start_ms))

    if first_in is not None and first_in.timestamp_ms <= t0:
        # In ground before the break. Did contact stay until the break?
        lost = [a for a in A if first_in.timestamp_ms < a.timestamp_ms < t1 and a.state != "IN"]
        definite_loss = [a for a in lost if a.state in ("NOT_IN_BEYOND", "AIRBORNE_BEHIND")]
        last_pre_break = [a for a in A if a.timestamp_ms < t1 and a.state != "UNKNOWN"][-1]
        if not definite_loss and (not lost or last_pre_break.state == "IN"):
            res.status = CreaseStatus.IN_GROUND
            res.reasons.append(f"Grounded behind the crease at frame {first_in.frame_index} ({first_in.timestamp_ms:.1f} ms), "
                               f"before the wicket was broken (intact at {t0:.1f} ms)")
            return res
        if not definite_loss:
            res.status = CreaseStatus.INCONCLUSIVE
            res.reasons.append("Grounded behind before the break, but the state in later frames before the break is unknown")
            return res
        loss = definite_loss[0]
        res.timeline.append(TimelineEvent(t_start_ms=loss.timestamp_ms, t_end_ms=loss.timestamp_ms, kind="contact_lost",
                                          description="Bat/person no longer grounded behind the crease", frames=[loss.frame_index]))
        if facts.running_or_diving_into_ground is True:
            res.status = CreaseStatus.IN_GROUND
            res.reasons.append(f"Grounded behind at frame {first_in.frame_index} before the break, then lost contact; "
                               "Law 30.1.2 applies (running or diving into the ground, as supplied)")
        elif facts.running_or_diving_into_ground is None:
            res.status = CreaseStatus.INCONCLUSIVE
            res.reasons.append(f"Grounded behind at frame {first_in.frame_index}, then lost contact at frame {loss.frame_index} "
                               "before the break: Law 30.1.2 protects the batter only if running or diving into the ground; "
                               "supply that fact")
        else:
            last_before = [a for a in A if a.timestamp_ms <= t0][-1]
            after = first(lambda a: a.timestamp_ms >= t1, A)
            if last_before.state in ("NOT_IN_BEYOND", "AIRBORNE_BEHIND") and after is not None and after.state == "NOT_IN_BEYOND":
                res.status = CreaseStatus.OUT_OF_GROUND
                res.reasons.append("Batter left their ground (Law 30.1.2 not applicable) and was out of ground across the break")
            else:
                res.status = CreaseStatus.INCONCLUSIVE
                res.reasons.append("Batter left their ground before the break, but the state at the break is not established")
        return res

    after = first(lambda a: a.timestamp_ms >= t1 and a.state != "UNKNOWN", A)
    if after is None:
        res.status = CreaseStatus.INCONCLUSIVE
        res.reasons.append("No usable observation at or after the first broken frame")
        return res
    if after.state == "NOT_IN_BEYOND":
        res.status = CreaseStatus.OUT_OF_GROUND
        m = after.measurement
        dist = (f"{-m.signed_m * 100:.1f} cm" if m and m.signed_m is not None else f"{-m.signed_px:.1f} px") if m else "?"
        res.reasons.append(f"Never observed grounded behind the crease before the break; at frame {after.frame_index} "
                           f"({after.timestamp_ms:.1f} ms, wicket already broken) the point is {dist} short of the crease")
        if after.timestamp_ms > t1:
            res.warnings.append(f"No observation exactly at the first broken frame; using frame {after.frame_index} "
                                f"({after.timestamp_ms - t1:.1f} ms later), assuming the batter did not ground and then move forward")
        return res
    if after.state == "IN":
        res.status = CreaseStatus.INCONCLUSIVE
        lo = res.grounding_window_ms[0] if res.grounding_window_ms else None
        lo_txt = f"{lo:.1f}" if lo is not None else "an unobserved time"
        res.reasons.append(f"Grounding happened between {lo_txt} and {after.timestamp_ms:.1f} ms and the wicket was broken between "
                           f"{t0:.1f} and {t1:.1f} ms: the windows overlap, so the order is not observable at this frame rate")
        return res
    res.status = CreaseStatus.INCONCLUSIVE
    res.reasons.append(f"At frame {after.frame_index} the point is behind the crease but not grounded; it may have been grounded "
                       "between frames (Law 30.1.2)")
    return res
