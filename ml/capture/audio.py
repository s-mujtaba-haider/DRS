"""Microphone capture on the shared host clock (e.g. a stump microphone for edge review).

Each audio block is stamped with the host time of its first sample. With PortAudio this uses the
stream's ADC timestamp (``inputBufferAdcTime``) translated to the host clock; when the host API
does not provide it (MME), the callback arrival time minus the block duration is used instead.
Blocks go to rolling 16-bit WAV segments with a JSONL sidecar ``{sample, n, host_ns}``.

On export, host time is fitted against sample index over the window (a straight line, robust to
callback jitter and accounting for audio-clock drift), so every exported sample has a time on the
same clock as the camera frames. The fit residual is reported as the timing uncertainty; the
microphone's own latency and the sound's travel time (~2.9 ms per metre) are *not* included.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.capture.sources import _sleep_until, clock_ns

log = logging.getLogger("drs.capture.audio")


class AudioSource:
    sample_rate: int
    channels: int = 1
    description: str = ""

    def open(self) -> None:  # pragma: no cover - interface
        pass

    def read(self, timeout: float = 1.0) -> tuple[np.ndarray, int] | None:  # pragma: no cover - interface
        """Next block as (float32 samples [n] mono, host_ns of the first sample); None on timeout."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass


def list_input_devices() -> list[dict]:
    """PortAudio input devices. WASAPI devices provide ADC timestamps (sub-millisecond alignment)."""
    try:
        import sounddevice as sd
    except Exception as exc:  # PortAudio missing
        return [{"error": f"sounddevice unavailable: {exc}"}]
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            out.append({"index": i, "name": d["name"], "host_api": api, "default_sample_rate": d["default_samplerate"],
                        "recommended": "WASAPI" in api})
    return out


class SoundDeviceSource(AudioSource):
    """A PortAudio input device (sounddevice). ``device`` is an index, a name, or None for the default."""

    def __init__(self, device: int | str | None = None, sample_rate: int = 48000, blocksize: int = 960):
        self.device, self.sample_rate, self.blocksize = device, int(sample_rate), int(blocksize)
        self.q: queue.Queue = queue.Queue(maxsize=2000)
        self.overflows = 0
        self.timestamp_basis = "adc"
        self.stream = None
        self.description = f"sounddevice:{device if device is not None else 'default'}@{sample_rate}"

    def _cb(self, indata, frames, time_info, status):
        now = clock_ns()
        if status and status.input_overflow:
            self.overflows += 1
        adc, cur = getattr(time_info, "inputBufferAdcTime", 0.0), getattr(time_info, "currentTime", 0.0)
        if adc and cur and 0 <= cur - adc < 1.0:
            first = now - int((cur - adc) * 1e9)
        else:
            self.timestamp_basis = "callback"
            first = now - int(frames / self.sample_rate * 1e9)
        x = indata[:, 0].astype(np.float32, copy=True)
        try:
            self.q.put_nowait((x, first))
        except queue.Full:
            self.overflows += 1

    def open(self) -> None:
        import sounddevice as sd

        self.stream = sd.InputStream(device=self.device, channels=1, samplerate=self.sample_rate, blocksize=self.blocksize,
                                     dtype="float32", callback=self._cb)
        self.stream.start()

    def read(self, timeout: float = 1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class SyntheticAudioSource(AudioSource):
    """Noise plus sharp 7 kHz clicks every ``click_every_s`` (host times kept in ``click_host_ns``), for tests."""

    def __init__(self, sample_rate: int = 48000, blocksize: int = 960, click_every_s: float = 0.5, jitter_ms: float = 2.0,
                 duration_s: float | None = None, seed: int = 0):
        self.sample_rate, self.blocksize, self.click_every_s = sample_rate, blocksize, click_every_s
        self.jitter_ms, self.duration_s = jitter_ms, duration_s
        self.rng = np.random.default_rng(seed)
        self.click_host_ns: list[int] = []
        self._t0: int | None = None
        self._k = 0
        self.overflows = 0
        self.timestamp_basis = "synthetic"
        self.description = f"synthetic-audio@{sample_rate}"

    def open(self) -> None:
        self._t0 = clock_ns()

    def read(self, timeout: float = 1.0):
        sr, n = self.sample_rate, self.blocksize
        if self.duration_s is not None and self._k >= self.duration_s * sr:
            return None
        first = self._t0 + int(self._k / sr * 1e9)
        _sleep_until(first + int(n / sr * 1e9))  # block is available once fully "recorded"
        x = (0.002 * self.rng.normal(size=n)).astype(np.float32)
        period = int(self.click_every_s * sr)
        for j in range(n):
            k = self._k + j
            if k % period == period // 2:
                m = min(n - j, int(0.003 * sr))
                x[j:j + m] += 0.5 * np.sin(2 * np.pi * 7000 * np.arange(m) / sr) * np.exp(-np.arange(m) / (0.0008 * sr))
                self.click_host_ns.append(self._t0 + int(k / sr * 1e9))
        self._k += n
        # report the first-sample time with callback-like jitter, as a real driver would
        return x, first + int(self.rng.normal(0, self.jitter_ms) * 1e6)


@dataclass
class AudioStats:
    blocks: int = 0
    samples: int = 0
    level_dbfs: float | None = None
    last_block_host_ns: int | None = None
    error: str | None = None
    segments: list[str] = field(default_factory=list)


class AudioRecorder:
    def __init__(self, source: AudioSource, directory: str | Path, segment_s: float = 10.0, retention_s: float = 300.0):
        self.source, self.dir = source, Path(directory)
        self.segment_s, self.retention_s = segment_s, retention_s
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stats = AudioStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.running = False

    def start(self) -> None:
        self.source.open()
        self.running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="drs-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.source.close()
        self.running = False

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        sr = self.source.sample_rate
        seg_idx, wav, side, seg_samples, seg_t0 = 0, None, None, 0, None
        try:
            while not self._stop.is_set():
                blk = self.source.read(timeout=0.5)
                if blk is None:
                    if self.source.__class__ is SyntheticAudioSource and self.source.duration_s is not None:
                        break
                    continue
                x, host_ns = blk
                if wav is None or (seg_t0 is not None and host_ns - seg_t0 > self.segment_s * 1e9):
                    if wav is not None:
                        wav.close()
                        side.close()
                    name = f"aseg_{seg_idx:06d}"
                    wav = wave.open(str(self.dir / f"{name}.wav"), "wb")
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sr)
                    side = (self.dir / f"{name}.jsonl").open("a")
                    seg_idx, seg_samples, seg_t0 = seg_idx + 1, 0, host_ns
                    self.stats.segments.append(name)
                    self._prune()
                wav.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
                side.write(json.dumps({"sample": seg_samples, "n": len(x), "host_ns": int(host_ns)}) + "\n")
                side.flush()
                seg_samples += len(x)
                rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
                self.stats.blocks += 1
                self.stats.samples += len(x)
                self.stats.level_dbfs = round(float(20 * np.log10(rms)), 1)
                self.stats.last_block_host_ns = host_ns
        except Exception as exc:  # reported in status, never crashes the server
            log.exception("audio recorder failed")
            self.stats.error = f"{type(exc).__name__}: {exc}"
        finally:
            if wav is not None:
                wav.close()
                side.close()
            self.running = False

    def _prune(self) -> None:
        keep = max(2, int(self.retention_s / self.segment_s) + 1)
        for name in self.stats.segments[:-keep]:
            for ext in (".wav", ".jsonl"):
                (self.dir / f"{name}{ext}").unlink(missing_ok=True)
        self.stats.segments = self.stats.segments[-keep:]

    def status(self) -> dict:
        st = self.stats
        return {"source": self.source.description, "running": self.running and self.is_alive(), "sample_rate": self.source.sample_rate,
                "blocks": st.blocks, "seconds": round(st.samples / self.source.sample_rate, 2), "level_dbfs": st.level_dbfs,
                "overflows": getattr(self.source, "overflows", 0), "timestamp_basis": getattr(self.source, "timestamp_basis", None),
                "last_block_age_s": None if st.last_block_host_ns is None else round((clock_ns() - st.last_block_host_ns) / 1e9, 3),
                "error": st.error}


def _read_blocks(directory: Path) -> list[tuple[Path, list[dict]]]:
    out = []
    for side in sorted(directory.glob("aseg_*.jsonl")):
        recs = [json.loads(line) for line in side.read_text().splitlines() if line.strip()]
        if recs:
            out.append((side.with_suffix(".wav"), recs))
    return out


def export_audio(directory: str | Path, t0_ns: int, t1_ns: int, out_path: str | Path, sample_rate: int) -> dict:
    """Write the samples with host time in [t0, t1] to ``out_path`` (16-bit WAV) with their timing fit."""
    directory = Path(directory)
    ks, hs, chunks, cum = [], [], [], 0
    pad_ns = int(2e9)
    for wav_path, recs in _read_blocks(directory):
        if recs[-1]["host_ns"] < t0_ns - pad_ns or recs[0]["host_ns"] > t1_ns + pad_ns:
            continue
        try:
            with wave.open(str(wav_path), "rb") as w:
                data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        except (OSError, EOFError, wave.Error):
            continue
        for r in recs:
            seg = data[r["sample"]: r["sample"] + r["n"]]
            if len(seg) < r["n"]:
                break
            ks.append(cum)
            hs.append(r["host_ns"])
            chunks.append(seg)
            cum += r["n"]
    if len(ks) < 3:
        return {"status": "no_audio_in_window"}
    k, h = np.array(ks, float), np.array(hs, float)
    span_s = (h[-1] - h[0]) / 1e9
    nominal_b = 1e9 / sample_rate
    if span_s < 10.0:
        # Audio clocks drift < ~100 ppm (< 1 ms over 10 s), while callback jitter makes a free slope
        # poorly determined over short spans: keep the nominal rate and fit the offset robustly.
        b = nominal_b
        a = float(np.median(h - b * k))
        res = h - (a + b * k)
        mad = float(np.median(np.abs(res))) * 1.4826 + 1.0
        keep = np.abs(res) <= 3 * mad + 1e5
        a = float(np.median(h[keep] - b * k[keep]))
    else:
        # Long spans: fit the rate too (drift matters), then refit without blocks > 3 MAD away.
        b, a = np.polyfit(k, h, 1)
        res = h - (a + b * k)
        mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826 + 1.0
        keep = np.abs(res) <= 3 * mad + 1e5
        b, a = np.polyfit(k[keep], h[keep], 1)
    res = h[keep] - (a + b * k[keep])
    samples = np.concatenate(chunks)
    s0 = int(np.ceil((t0_ns - a) / b))
    s1 = int(np.floor((t1_ns - a) / b))
    s0, s1 = max(0, s0), min(len(samples), s1)
    if s1 - s0 < sample_rate * 0.05:
        return {"status": "no_audio_in_window"}
    out_path = Path(out_path)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples[s0:s1].tobytes())
    resid_ms = float(np.std(res)) / 1e6
    n_used = int(keep.sum())
    ratio = (1e9 / b) / sample_rate
    return {"status": "ok", "path": str(out_path), "sample_rate": sample_rate, "n_samples": int(s1 - s0),
            "start_host_ns": int(round(a + b * s0)), "clock_ratio": round(float(ratio), 7),
            "fit_residual_ms": round(resid_ms, 3),
            # offset uncertainty: residual scatter averaged over the blocks used, plus one sample
            "uncertainty_ms": round(float(2 * resid_ms / np.sqrt(max(n_used, 1)) + 1000 / sample_rate), 3),
            "rate_model": "nominal" if span_s < 10.0 else "fitted",
            "note": "Sample times are on the shared host clock (fit of block timestamps). Microphone/ADC latency "
                    "beyond the driver's timestamp and sound travel time (~2.9 ms per metre) are not included."}
