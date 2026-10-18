"""Camera time-offset estimation from shared audio or matched visual events.

Convention: ``offset_ms`` maps camera B's video timestamps onto camera A's
clock, ``t_a = t_b + offset_ms``. Timestamps are those produced by
``ml.video.ingest.iter_frames`` (milliseconds since the first video frame).
Applied to a ``CameraModel`` (``time_offset_ms`` = "add to this camera's
timestamps to get the common clock"), camera A is the reference.

Neither method gives hardware-grade synchronisation:

* Audio aligns the *audio* tracks. The audio-to-video offset inside each file
  (capture and mux latency) is unknown, and sound travels about 34 cm per
  millisecond, so microphones at different distances from the source differ.
* Visual events are only known to lie between two frames, so the offset is an
  interval whose width is set by the frame intervals.
"""

from __future__ import annotations

import math
from typing import Literal

import av
import numpy as np
from pydantic import BaseModel, Field

from ml.calibration.camera import CameraModel

SPEED_OF_SOUND_M_S = 343.0

SyncStatus = Literal["OK", "LOW_CONFIDENCE", "NO_AUDIO", "INCONSISTENT", "INSUFFICIENT_DATA"]


class SyncResult(BaseModel):
    method: Literal["audio_gcc_phat", "visual_events"]
    status: SyncStatus
    reference: str = "a"
    offset_ms: float | None = Field(None, description="t_a = t_b + offset_ms")
    uncertainty_ms: float | None = Field(None, description="half-width of the ~95% interval (audio) or feasible interval (events)")
    feasible_interval_ms: tuple[float, float] | None = None
    confidence: float | None = Field(None, description="audio: peak-to-sidelobe ratio (not a probability)")
    n_windows: int | None = None
    window_offsets_ms: list[float] = Field(default_factory=list)
    n_events: int | None = None
    event_intervals_ms: list[tuple[float, float]] = Field(default_factory=list)
    sample_rate_hz: int | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------- audio
def load_audio(path: str, sample_rate: int = 16000, max_duration_s: float | None = 600.0) -> tuple[np.ndarray, float] | None:
    """Mono float32 audio and its start time (ms) on the video-timestamp axis; None without audio.

    The start time is the first audio PTS minus the first video PTS (0 when the file has no video),
    so audio sample k is at ``start_ms + 1000 * k / sample_rate`` in iter_frames' time axis.
    """
    container = av.open(str(path))
    try:
        if not container.streams.audio:
            return None
        astream = container.streams.audio[0]
        v0_ms = 0.0
        if container.streams.video:
            vs = container.streams.video[0]
            for fr in container.decode(vs):
                if fr.pts is not None:
                    v0_ms = float(fr.pts * fr.time_base) * 1000.0
                break
            container.seek(0)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        chunks, a0_ms, n = [], None, 0
        limit = int(max_duration_s * sample_rate) if max_duration_s else None
        for fr in container.decode(astream):
            if a0_ms is None and fr.pts is not None:
                a0_ms = float(fr.pts * fr.time_base) * 1000.0
            for out in resampler.resample(fr):
                arr = out.to_ndarray().reshape(-1)
                chunks.append(arr)
                n += len(arr)
            if limit and n >= limit:
                break
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
        if not chunks:
            return None
        x = np.concatenate(chunks).astype(np.float32)
        if limit:
            x = x[:limit]
        return x, (a0_ms or 0.0) - v0_ms
    finally:
        container.close()


def _gcc_phat(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Generalised cross-correlation (PHAT). cc[lag] is large when a[n] ~ b[n - lag]."""
    n = len(a) + len(b)
    nfft = 1 << (n - 1).bit_length()
    A, B = np.fft.rfft(a, nfft), np.fft.rfft(b, nfft)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    cc = np.fft.irfft(R, nfft)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    return np.arange(-max_lag, max_lag + 1), cc


def _peak(lags: np.ndarray, cc: np.ndarray, exclude: int) -> tuple[float, float]:
    """Sub-sample peak lag (parabolic) and peak-to-sidelobe ratio."""
    i = int(np.argmax(cc))
    frac = 0.0
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    mask = np.ones(len(cc), bool)
    mask[max(0, i - exclude): i + exclude + 1] = False
    side = cc[mask]
    psr = float((cc[i] - side.mean()) / (side.std() + 1e-12)) if side.size > 10 else 0.0
    return float(lags[i] + frac), psr


def audio_offset_from_signals(a: np.ndarray, b: np.ndarray, sample_rate: int, a_start_ms: float = 0.0,
                              b_start_ms: float = 0.0, max_offset_ms: float = 2000.0, n_windows: int = 6,
                              min_psr: float = 8.0, max_window_spread_ms: float | None = None) -> SyncResult:
    """Offset between two audio signals with known start times on their video time axes."""
    warnings = ["Audio alignment is not video-frame alignment: each file's audio-to-video mux offset is unknown",
                f"Sound travels {SPEED_OF_SOUND_M_S / 1000:.3f} m per ms; microphone distance differences bias the offset"]
    max_lag = int(math.ceil(max_offset_ms * sample_rate / 1000.0)) + int(abs(a_start_ms - b_start_ms) * sample_rate / 1000) + 1
    if min(len(a), len(b)) < sample_rate // 4:
        return SyncResult(method="audio_gcc_phat", status="INSUFFICIENT_DATA", sample_rate_hz=sample_rate,
                          warnings=[*warnings, "Less than 0.25 s of audio"])
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return SyncResult(method="audio_gcc_phat", status="LOW_CONFIDENCE", sample_rate_hz=sample_rate,
                          warnings=[*warnings, "Audio track is silent"])
    exclude = max(3, sample_rate // 1000)

    def lag_to_offset(lag: float) -> float:
        # a[n] ~ b[n - lag]: an event at sample m in b sits at m + lag in a.
        # t_a(event) = a_start + (m + lag)/sr, t_b(event) = b_start + m/sr  =>  offset = t_a - t_b.
        return a_start_ms - b_start_ms + 1000.0 * lag / sample_rate

    lags, cc = _gcc_phat(a, b, max_lag)
    lag, psr = _peak(lags, cc, exclude)
    offset = lag_to_offset(lag)

    # Consistency: correlate sub-windows of A with all of B; a window "agrees" when its own
    # peak lands within 10 ms of the global estimate. Silent or ambiguous windows disagree.
    win_offsets, n_tried = [], 0
    n = len(a)
    agree_tol_ms = 10.0
    if n_windows > 1 and n >= n_windows * sample_rate // 2:
        wlen = n // n_windows
        for k in range(n_windows):
            seg = a[k * wlen: (k + 1) * wlen]
            if np.std(seg) < 1e-6:
                continue
            n_tried += 1
            l2, cc2 = _gcc_phat(seg, b, max_lag + n)
            lk, _ = _peak(l2, cc2, exclude)
            ok = lag_to_offset(lk + k * wlen)
            if abs(ok - offset) <= agree_tol_ms:
                win_offsets.append(ok)
    quant = 1000.0 / sample_rate / math.sqrt(12)
    spread = float(np.std(win_offsets)) if len(win_offsets) >= 3 else None
    if spread is not None:
        unc = 2.0 * math.sqrt(spread**2 + quant**2)
    else:
        unc = 2.0 * quant
        warnings.append("Too few agreeing sub-windows to measure consistency; uncertainty is the sampling floor only")
    status: SyncStatus = "OK" if psr >= min_psr else "LOW_CONFIDENCE"
    if status != "OK":
        warnings.append(f"Correlation peak is weak (peak-to-sidelobe ratio {psr:.1f} < {min_psr})")
    if n_tried >= 3 and len(win_offsets) < 0.5 * n_tried:
        status = "LOW_CONFIDENCE"
        warnings.append(f"Only {len(win_offsets)} of {n_tried} audio sub-windows agree with the global offset")
    if max_window_spread_ms is not None and spread is not None and spread > max_window_spread_ms:
        status = "LOW_CONFIDENCE"
        warnings.append(f"Sub-window offsets disagree (std {spread:.2f} ms)")
    if abs(offset - (a_start_ms - b_start_ms)) > max_offset_ms * 0.98:
        status = "LOW_CONFIDENCE"
        warnings.append("Peak at the edge of the search window; true offset may be outside it")
    return SyncResult(method="audio_gcc_phat", status=status, offset_ms=offset, uncertainty_ms=unc, confidence=psr,
                      n_windows=len(win_offsets), window_offsets_ms=win_offsets, sample_rate_hz=sample_rate,
                      feasible_interval_ms=(offset - unc, offset + unc), warnings=warnings)


def audio_offset(video_a: str, video_b: str, sample_rate: int = 16000, max_offset_ms: float = 2000.0,
                 max_duration_s: float = 120.0, n_windows: int = 6, min_psr: float = 8.0) -> SyncResult:
    """Estimate the video-timestamp offset between two recordings from their audio (GCC-PHAT)."""
    la = load_audio(video_a, sample_rate, max_duration_s)
    lb = load_audio(video_b, sample_rate, max_duration_s)
    if la is None or lb is None:
        missing = [n for n, x in (("a", la), ("b", lb)) if x is None]
        return SyncResult(method="audio_gcc_phat", status="NO_AUDIO", sample_rate_hz=sample_rate,
                          warnings=[f"No audio track in file(s) {', '.join(missing)}"])
    (a, a0), (b, b0) = la, lb
    return audio_offset_from_signals(a, b, sample_rate, a0, b0, max_offset_ms, n_windows, min_psr)


# ------------------------------------------------------------- visual events
EventSpec = float | tuple[float, float]


def _event_interval(ev: EventSpec, fps: float | None, mode: str) -> tuple[float, float]:
    if isinstance(ev, (tuple, list)):
        lo, hi = float(ev[0]), float(ev[1])
        if hi < lo:
            raise ValueError("event interval must be (last_frame_before, first_frame_after)")
        return lo, hi
    if not fps or fps <= 0:
        raise ValueError("fps is required for single-timestamp events")
    dt = 1000.0 / fps
    t = float(ev)
    if mode == "first_seen":
        return t - dt, t  # happened after the previous frame, at or before this one
    return t - dt / 2, t + dt / 2  # "nearest": the frame closest to the event


def event_offset(events_a: list[EventSpec], events_b: list[EventSpec], fps_a: float | None = None,
                 fps_b: float | None = None, mode: Literal["first_seen", "nearest"] = "first_seen") -> SyncResult:
    """Offset from the same physical events seen by both cameras.

    Each event is either a timestamp (ms) of the frame where it is first visible (``mode=
    "first_seen"``: the event lies between the previous frame and this one) or of the nearest frame
    (``"nearest"``: within half a frame interval), or an explicit ``(last_before, first_after)``
    interval. Every event constrains ``offset = t_a - t_b`` to an interval; the result is their
    intersection. An empty intersection means the events were mismatched or the clocks drift:
    status INCONSISTENT.
    """
    if len(events_a) != len(events_b):
        raise ValueError("events_a and events_b must be matched one-to-one")
    if not events_a:
        return SyncResult(method="visual_events", status="INSUFFICIENT_DATA", n_events=0,
                          warnings=["No matched events"])
    intervals = []
    for ea, eb in zip(events_a, events_b):
        la, ha = _event_interval(ea, fps_a, mode)
        lb, hb = _event_interval(eb, fps_b, mode)
        intervals.append((la - hb, ha - lb))
    lo = max(i[0] for i in intervals)
    hi = min(i[1] for i in intervals)
    warnings = ["Assumes no clock drift between cameras over the span of the events"]
    if len(intervals) == 1:
        warnings.append("Single event: no consistency check possible")
    if lo > hi:
        mids = [(i[0] + i[1]) / 2 for i in intervals]
        return SyncResult(method="visual_events", status="INCONSISTENT", offset_ms=float(np.median(mids)),
                          uncertainty_ms=None, n_events=len(intervals), event_intervals_ms=intervals,
                          warnings=[*warnings, f"Event constraints do not overlap (gap {lo - hi:.2f} ms): mismatched events, "
                                               "wrong frame choice, or clock drift"])
    return SyncResult(method="visual_events", status="OK", offset_ms=(lo + hi) / 2, uncertainty_ms=(hi - lo) / 2,
                      feasible_interval_ms=(lo, hi), n_events=len(intervals), event_intervals_ms=intervals,
                      warnings=warnings)


# ------------------------------------------------------------------ applying
def apply_sync(camera_b: CameraModel, sync: SyncResult, reference_offset_ms: float = 0.0,
               reference_uncertainty_ms: float = 0.0) -> CameraModel:
    """Copy of camera B with ``time_offset_ms`` / ``sync_uncertainty_ms`` set from a SyncResult.

    ``reference_offset_ms`` is camera A's own offset to the common clock (0 when A is the reference).
    Refuses results without a usable offset; LOW_CONFIDENCE results are applied with a warning.
    """
    if sync.offset_ms is None or sync.status in ("NO_AUDIO", "INCONSISTENT", "INSUFFICIENT_DATA"):
        raise ValueError(f"Cannot apply a sync result with status {sync.status}")
    cam = camera_b.model_copy(deep=True)
    cam.time_offset_ms = reference_offset_ms + sync.offset_ms
    unc = sync.uncertainty_ms if sync.uncertainty_ms is not None else float("inf")
    cam.sync_uncertainty_ms = math.sqrt(unc**2 + reference_uncertainty_ms**2)
    cam.warnings.append(f"Time offset from {sync.method} ({sync.status}); software synchronisation, not hardware trigger")
    if sync.status == "LOW_CONFIDENCE":
        cam.warnings.append("Synchronisation estimate has LOW_CONFIDENCE: treat multi-view reconstruction as degraded")
    return cam
