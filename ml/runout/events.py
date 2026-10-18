"""Automatic wicket-break *candidate* detection from video frames (vision layer).

Law 29.1 (MCC Laws, 2017 Code 4th Edition, 2026): the wicket is broken when at least
one bail is completely removed from the top of the stumps, or one or more stumps is
removed from the ground. Law 29.2.2: a temporary disturbance of a bail is not removal.

Method: a fixed region of interest (ROI) around the stumps (from the operator or the
stumps detector) is compared with a reference built from the first frames, which must
show the intact wicket. A pixel counts as changed when it differs from the reference by
more than a threshold set from the noise between the reference frames. Two cues:

* ``roi_changed_frac``: fraction of changed pixels in the whole ROI (a stump moved);
* ``bails_changed_frac``: fraction of changed pixels in the bails band at the top of the ROI.

A frame is flagged when either cue exceeds its threshold. ``persist_frames`` consecutive
flagged frames are required, so a ball or glove passing through the ROI for a frame or two
is not reported.

The output is a candidate interval only: the moment of breaking lies between the last
intact frame and the first broken frame. Gloves, the ball, the batter and camera motion
also change the ROI. A person must confirm visually that a bail was completely removed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

ENGINE_VERSION = "wicket-break-0.1.0"


class WicketBreakFrameScore(BaseModel):
    frame_index: int
    timestamp_ms: float
    roi_changed_frac: float
    bails_changed_frac: float
    flagged: bool


class WicketBreakDetection(BaseModel):
    engine_version: str = ENGINE_VERSION
    status: Literal["BREAK_CANDIDATE", "NO_CHANGE_DETECTED", "INSUFFICIENT_DATA"]
    roi: tuple[int, int, int, int]
    last_intact_frame: int | None = None
    last_intact_ms: float | None = None
    first_broken_frame: int | None = None
    first_broken_ms: float | None = None
    interval_ms: float | None = Field(None, description="width of the window in which the wicket was broken")
    score: float | None = None
    pixel_threshold: float | None = Field(None, description="grey-level difference counted as a change")
    roi_frac_threshold: float | None = None
    bails_frac_threshold: float | None = None
    frames: list[WicketBreakFrameScore] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=lambda: [
        "Candidate only: gloves, ball, batter or camera motion also change the stumps region; confirm a bail "
        "was completely removed (Law 29.1)"])


def _unpack(f: Any) -> tuple[int, float, np.ndarray]:
    if hasattr(f, "image"):
        return int(f.index), float(f.timestamp_ms), f.image
    i, t, img = f
    return int(i), float(t), img


def _crop(img: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g[y1:y2, x1:x2], (3, 3), 0).astype(np.float32)


def _clamp_roi(roi, shape) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in roi)
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError("ROI must be at least 8x8 px inside the frame")
    return x1, y1, x2, y2


def detect_wicket_break(
    frames: Iterable,
    roi: tuple[float, float, float, float],
    *,
    n_reference: int = 5,
    bails_band_frac: float = 0.3,
    persist_frames: int = 3,
    min_pixel_diff: float = 25.0,
    min_roi_frac: float = 0.03,
    min_bails_frac: float = 0.02,
    noise_k: float = 6.0,
) -> WicketBreakDetection:
    """Find the first persistent change of the stumps ROI.

    ``frames``: iterable of ``FrameRecord`` (``ml.video.ingest``) or ``(index, timestamp_ms, image)``.
    ``roi``: x1, y1, x2, y2 in pixels around the stumps *and bails*, with the bails in the top band.
    The first ``n_reference`` frames must show the intact wicket.
    """
    ref_frames: list[np.ndarray] = []
    ref_meta: list[tuple[int, float]] = []
    scores: list[WicketBreakFrameScore] = []
    roi_c: tuple[int, int, int, int] | None = None
    ref = None
    band = 0
    pix_thr = roi_thr = bails_thr = 0.0
    run: list[int] = []
    found: int | None = None  # index into scores of the first flagged frame of the persistent run

    def fracs(g: np.ndarray) -> tuple[float, float]:
        changed = np.abs(g - ref) > pix_thr
        return float(changed.mean()), float(changed[:band].mean())

    for f in frames:
        i, t, img = _unpack(f)
        if roi_c is None:
            roi_c = _clamp_roi(roi, img.shape)
        g = _crop(img, roi_c)
        if ref is None:
            ref_frames.append(g)
            ref_meta.append((i, t))
            if len(ref_frames) == n_reference:
                ref = np.median(np.stack(ref_frames), axis=0)
                band = max(2, int(round(ref.shape[0] * bails_band_frac)))
                resid = np.concatenate([(r - ref).ravel() for r in ref_frames])
                pix_noise = 1.4826 * float(np.median(np.abs(resid)))
                pix_thr = max(min_pixel_diff, noise_k * pix_noise)
                nf = [fracs(r) for r in ref_frames]
                roi_thr = max(min_roi_frac, noise_k * max(a for a, _ in nf))
                bails_thr = max(min_bails_frac, noise_k * max(b for _, b in nf))
                for (ri, rt) in ref_meta:
                    scores.append(WicketBreakFrameScore(frame_index=ri, timestamp_ms=rt, roi_changed_frac=0.0,
                                                        bails_changed_frac=0.0, flagged=False))
            continue
        a, b = fracs(g)
        flagged = a > roi_thr or b > bails_thr
        scores.append(WicketBreakFrameScore(frame_index=i, timestamp_ms=t, roi_changed_frac=round(a, 4),
                                            bails_changed_frac=round(b, 4), flagged=flagged))
        if found is None:
            if flagged:
                run.append(len(scores) - 1)
                if len(run) >= persist_frames:
                    found = run[0]
            else:
                run = []

    det = WicketBreakDetection(status="INSUFFICIENT_DATA", roi=roi_c or tuple(int(v) for v in roi), frames=scores,
                               pixel_threshold=pix_thr or None, roi_frac_threshold=roi_thr or None,
                               bails_frac_threshold=bails_thr or None)
    if ref is None:
        det.reasons.append(f"Fewer than {n_reference} frames: no reference of the intact wicket")
        return det
    if found is None:
        det.status = "NO_CHANGE_DETECTED"
        det.reasons.append(f"No change persisted for {persist_frames} frames above the noise thresholds")
        if run:
            det.reasons.append("A change starts in the final frames but does not persist long enough; extend the clip")
        return det
    fb, li = scores[found], scores[found - 1]
    det.status = "BREAK_CANDIDATE"
    det.first_broken_frame, det.first_broken_ms = fb.frame_index, fb.timestamp_ms
    det.last_intact_frame, det.last_intact_ms = li.frame_index, li.timestamp_ms
    det.interval_ms = fb.timestamp_ms - li.timestamp_ms
    det.score = max(fb.roi_changed_frac / roi_thr, fb.bails_changed_frac / bails_thr)
    det.reasons.append(f"Stumps region changed persistently from frame {fb.frame_index}; the wicket was intact at "
                       f"frame {li.frame_index}, so the break lies within a {det.interval_ms:.1f} ms window")
    return det


def detect_wicket_break_in_video(path, roi, start_frame: int = 0, end_frame: int | None = None, **kw) -> WicketBreakDetection:
    from ml.video.ingest import iter_frames

    return detect_wicket_break(iter_frames(path, start_frame=start_frame, end_frame=end_frame), roi, **kw)
