"""Experimental audio transient analysis for edge (bat-ball contact) review.

Finds short, sharp, high-frequency sounds in a video's audio track and places them on the video's
timestamp axis, for review next to the frames (the role an "edge detector" plays in professional
DRS). A transient is only a *candidate*: bat on pad, bat on ground, ball on pad or glove, shoe
squeaks, stump-mic handling and crowd noise all produce one, and a faint edge can be inaudible.

Timing caveats, all reported as warnings:
* the audio/video mux offset of an ordinary camera or broadcast file is unknown;
* sound travels about 0.34 m per millisecond, so a microphone 10 m from the bat hears it ~29 ms late;
* lossy audio codecs (AAC/MP3) remove much of the content above ~16 kHz.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy.signal import butter, sosfiltfilt

ENGINE_VERSION = "audio-contact-0.1.0"
SPEED_OF_SOUND_MPS = 343.0


class AudioTransient(BaseModel):
    t_ms: float = Field(description="onset time on the video timestamp axis (before any mic-distance correction)")
    snr_db: float = Field(description="peak high-band energy above the local noise floor")
    rise_ms: float = Field(description="10%-90% rise time of the energy envelope; edges are sharp (a few ms)")
    duration_ms: float
    nearest_frame_index: int | None = None
    nearest_frame_ms: float | None = None


class AudioContactResult(BaseModel):
    engine_version: str = ENGINE_VERSION
    status: str  # TRANSIENTS_FOUND | NO_TRANSIENT | NO_AUDIO | INSUFFICIENT_DATA
    sample_rate_hz: int | None = None
    highpass_hz: float | None = None
    window_ms: tuple[float, float] | None = None
    noise_floor_db: float | None = None
    transients: list[AudioTransient] = Field(default_factory=list)
    envelope: dict | None = Field(None, description="{t0_ms, dt_ms, db: [...]} high-band energy for plotting")
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=lambda: [
        "Bat hitting pad or ground", "Ball hitting pad, glove or body", "Shoe or spike noise", "Stump-microphone handling",
        "Crowd, PA or commentary", "Audio compression artifacts"])


def _frames_db(x: np.ndarray, sr: int, frame_ms: float) -> tuple[np.ndarray, float]:
    n = max(1, int(round(sr * frame_ms / 1000)))
    m = len(x) // n
    e = (x[: m * n].reshape(m, n) ** 2).mean(1)
    return 10 * np.log10(e + 1e-12), n / sr * 1000


def analyze_audio_transients(
    x: np.ndarray,
    sample_rate: int,
    audio_start_ms: float,
    window_ms: tuple[float, float] | None = None,
    highpass_hz: float = 2000.0,
    min_snr_db: float = 15.0,
    max_rise_ms: float = 4.0,
    merge_ms: float = 15.0,
    frame_times_ms: list[tuple[int, float]] | None = None,
    source_rate: int | None = None,
    timing_note: str | None = None,
) -> AudioContactResult:
    """Transient candidates in mono audio ``x`` whose first sample is at ``audio_start_ms`` (video axis)."""
    res = AudioContactResult(status="INSUFFICIENT_DATA", sample_rate_hz=int(sample_rate), highpass_hz=highpass_hz)
    t_all = audio_start_ms + np.arange(len(x)) * 1000.0 / sample_rate
    if window_ms is not None:
        keep = (t_all >= window_ms[0]) & (t_all <= window_ms[1])
        x, t_all = x[keep], t_all[keep]
    if len(x) < sample_rate * 0.05:
        res.reasons.append("Less than 50 ms of audio in the window")
        return res
    res.window_ms = (float(t_all[0]), float(t_all[-1]))
    src = source_rate or sample_rate
    nyq = sample_rate / 2
    if highpass_hz >= 0.4 * src:
        highpass_hz = 0.2 * src
        res.highpass_hz = highpass_hz
        res.warnings.append(f"Source audio sampled at {src} Hz: high-pass lowered to {highpass_hz:.0f} Hz")
    if highpass_hz >= 0.8 * nyq:
        highpass_hz = 0.4 * nyq
        res.warnings.append(f"Audio sampled at {sample_rate} Hz: high-pass lowered to {highpass_hz:.0f} Hz; edge content is mostly above it")
        res.highpass_hz = highpass_hz
    if src < 32000:
        res.warnings.append("Sample rate below 32 kHz: the sharp high-frequency part of a fine edge is not recorded")
    sos = butter(4, highpass_hz / nyq, btype="highpass", output="sos")
    y = sosfiltfilt(sos, x.astype(np.float64))
    db, dt = _frames_db(y, sample_rate, 0.5)
    floor = float(np.median(db))
    mad = float(np.median(np.abs(db - floor))) * 1.4826 + 1e-6
    res.noise_floor_db = round(floor, 2)
    thr = floor + max(min_snr_db, 4 * mad)
    above = db > thr
    events: list[tuple[int, int]] = []
    i = 0
    while i < len(db):
        if above[i]:
            j = i
            while j + 1 < len(db) and (above[j + 1] or (j + 2 < len(db) and above[j + 2])):
                j += 1
            events.append((i, j))
            i = j + 1
        else:
            i += 1
    merged: list[tuple[int, int]] = []
    for a, b in events:
        if merged and (a - merged[-1][1]) * dt < merge_ms:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    t0 = float(t_all[0])
    for a, b in merged:
        seg = db[a: b + 1]
        pk = a + int(np.argmax(seg))
        lo = max(0, a - int(10 / dt))
        base = float(np.median(db[lo: a])) if a > lo else floor
        amp = db[pk] - base
        # Rise time: walk back from the peak to the last 10% crossing (an earlier noise blip must not
        # count as the onset), then forward to the first 90% crossing.
        j10 = pk
        while j10 > lo and db[j10 - 1] >= base + 0.1 * amp:
            j10 -= 1
        j90 = j10
        while j90 < pk and db[j90] < base + 0.9 * amp:
            j90 += 1
        rise = max(0.0, (j90 - j10) * dt)
        onset_ms = t0 + j10 * dt
        if rise > max_rise_ms:
            continue  # slow build-up: speech, crowd or wind rather than an impact
        tr = AudioTransient(t_ms=round(onset_ms, 2), snr_db=round(float(db[pk] - floor), 1), rise_ms=round(rise, 2),
                            duration_ms=round((b - a + 1) * dt, 2))
        if frame_times_ms:
            fi, ft = min(frame_times_ms, key=lambda f: abs(f[1] - onset_ms))
            tr.nearest_frame_index, tr.nearest_frame_ms = fi, round(ft, 3)
        res.transients.append(tr)
    step = max(1, len(db) // 2000)
    res.envelope = {"t0_ms": round(t0, 3), "dt_ms": dt * step, "db": [round(float(v), 1) for v in db[::step]]}
    res.status = "TRANSIENTS_FOUND" if res.transients else "NO_TRANSIENT"
    res.reasons.append(f"{len(res.transients)} sharp transient(s) at least {min_snr_db:.0f} dB above the "
                       f"{highpass_hz:.0f} Hz+ noise floor with rise time <= {max_rise_ms} ms")
    if not res.transients:
        res.reasons.append("No transient does not rule out a fine edge (inaudible, masked, or filtered by compression)")
    res.warnings += [
        timing_note or "Audio/video mux offset of this file is unknown: audio times may be shifted relative to the frames",
        f"Sound travels {SPEED_OF_SOUND_MPS / 1000:.2f} m per ms: a microphone d metres from the bat hears contact d/0.343 ms late",
        "A transient is a candidate sound, not proof of bat-ball contact",
    ]
    return res


def analyze_video_audio(path, center_ms: float | None = None, half_window_ms: float = 400.0, sample_rate: int = 48000,
                        frame_times_ms: list[tuple[int, float]] | None = None, **kw) -> AudioContactResult:
    import av

    from ml.synchronization.sync import load_audio

    with av.open(str(path)) as c:
        src = c.streams.audio[0].rate if c.streams.audio else None
    loaded = load_audio(str(path), sample_rate=sample_rate, max_duration_s=600)
    if loaded is None or src is None:
        return AudioContactResult(status="NO_AUDIO", reasons=["The video has no audio track"])
    x, a0 = loaded
    kw.setdefault("source_rate", int(src))
    win = (center_ms - half_window_ms, center_ms + half_window_ms) if center_ms is not None else None
    return analyze_audio_transients(x, sample_rate, a0, win, frame_times_ms=frame_times_ms, **kw)
