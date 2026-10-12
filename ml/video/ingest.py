"""Video ingestion: metadata probing and incremental frame decoding.

PyAV (bundled FFmpeg libraries) is used so per-frame presentation timestamps
are available. Frame indices are decode-order counters; timestamps come from
container PTS when present and are otherwise *estimated* from the nominal
frame rate and flagged as such.
"""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from ml.common.schemas import TimestampSource, VideoMetadata

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
VFR_RELATIVE_STD_THRESHOLD = 0.05


class VideoDecodeError(RuntimeError):
    """Raised when a file cannot be opened or contains no decodable video."""


@dataclass
class FrameRecord:
    index: int
    timestamp_ms: float
    timestamp_source: TimestampSource
    image: np.ndarray  # BGR uint8, full resolution unless resized by caller


def _open(path: str | os.PathLike) -> av.container.InputContainer:
    p = Path(path)
    if not p.is_file():
        raise VideoDecodeError(f"File not found: {p.name}")
    if p.stat().st_size == 0:
        raise VideoDecodeError("File is empty")
    try:
        container = av.open(str(p))
    except (av.error.FFmpegError, OSError) as exc:
        raise VideoDecodeError(f"Cannot open video: {exc}") from exc
    if not container.streams.video:
        container.close()
        raise VideoDecodeError("No video stream found")
    return container


def probe_video(path: str | os.PathLike, sample_frames: int = 240, full_scan: bool = False) -> VideoMetadata:
    """Inspect metadata and sample PTS deltas to detect variable frame rate.

    With ``full_scan`` every frame is decoded to count frames exactly.
    """
    p = Path(path)
    container = _open(p)
    warnings: list[str] = []
    try:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        width, height = int(ctx.width or 0), int(ctx.height or 0)
        avg_rate = stream.average_rate
        nominal_fps = float(avg_rate) if avg_rate else None
        if nominal_fps is None:
            warnings.append("Container does not declare an average frame rate")
        declared = int(stream.frames) if stream.frames else None
        if declared is None:
            warnings.append("Container does not declare a frame count")
        duration_s = None
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_s = container.duration / 1_000_000
        else:
            warnings.append("Duration not available from metadata")

        pts: list[float] = []
        decoded = 0
        decode_errors = 0
        limit = None if full_scan else sample_frames
        try:
            for frame in container.decode(stream):
                if frame.pts is not None and frame.time_base is not None:
                    pts.append(float(frame.pts * frame.time_base) * 1000.0)
                decoded += 1
                if not width:
                    width, height = frame.width, frame.height
                if limit is not None and decoded >= limit:
                    break
        except av.error.FFmpegError as exc:
            decode_errors += 1
            warnings.append(f"Decoding error after {decoded} frames: {exc}")

        if decoded == 0:
            raise VideoDecodeError("Video stream contains no decodable frames")

        vfr = None
        dmean = dstd = None
        if len(pts) >= 3:
            deltas = np.diff(np.array(sorted(pts)))
            deltas = deltas[deltas > 0]
            if deltas.size:
                dmean, dstd = float(deltas.mean()), float(deltas.std())
                vfr = bool(dstd / dmean > VFR_RELATIVE_STD_THRESHOLD)
                if vfr:
                    warnings.append(
                        "Variable frame rate detected: use per-frame timestamps, not index/fps"
                    )
                if nominal_fps and abs(1000.0 / dmean - nominal_fps) / nominal_fps > 0.05:
                    warnings.append(
                        f"Declared fps {nominal_fps:.3f} differs from measured PTS rate {1000.0 / dmean:.3f}"
                    )
        else:
            warnings.append("Too few PTS values to assess frame timing")
        if len(pts) < decoded:
            warnings.append("Some frames lack PTS; their timestamps will be estimated")

        return VideoMetadata(
            path_name=p.name,
            container_format=container.format.name if container.format else None,
            codec=ctx.name,
            width=width,
            height=height,
            nominal_fps=nominal_fps,
            frame_count_declared=declared,
            frame_count_decoded=decoded if (full_scan and decode_errors == 0) else None,
            duration_s=duration_s,
            time_base=str(stream.time_base) if stream.time_base else None,
            variable_frame_rate=vfr,
            pts_delta_ms_mean=dmean,
            pts_delta_ms_std=dstd,
            has_audio=bool(container.streams.audio),
            size_bytes=p.stat().st_size,
            warnings=warnings,
        )
    finally:
        container.close()


def iter_frames(
    path: str | os.PathLike,
    start_frame: int = 0,
    end_frame: int | None = None,
    step: int = 1,
    max_frames: int | None = None,
    resize_width: int | None = None,
) -> Iterator[FrameRecord]:
    """Yield frames incrementally (never loads the whole video into RAM)."""
    if step < 1:
        raise ValueError("step must be >= 1")
    container = _open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else None
        first_pts_ms: float | None = None
        emitted = 0
        for index, frame in enumerate(container.decode(stream)):
            if end_frame is not None and index >= end_frame:
                break
            if frame.pts is not None and frame.time_base is not None:
                ts = float(frame.pts * frame.time_base) * 1000.0
                if first_pts_ms is None:
                    first_pts_ms = ts
                ts_ms, src = ts - first_pts_ms, TimestampSource.CONTAINER_PTS
            elif fps:
                ts_ms, src = index * 1000.0 / fps, TimestampSource.ESTIMATED_FROM_FPS
            else:
                raise VideoDecodeError("Frame has no PTS and stream declares no frame rate")
            if index < start_frame or (index - start_frame) % step:
                continue
            img = frame.to_ndarray(format="bgr24")
            if resize_width and img.shape[1] > resize_width:
                import cv2

                h = int(round(img.shape[0] * resize_width / img.shape[1]))
                img = cv2.resize(img, (resize_width, h), interpolation=cv2.INTER_AREA)
            yield FrameRecord(index, ts_ms, src, img)
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break
    finally:
        container.close()


def extract_frames(
    path: str | os.PathLike,
    out_dir: str | os.PathLike,
    step: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> Path:
    """Write frames to disk with an ``index.csv`` preserving index and timestamp."""
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    index_path = out / "index.csv"
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if image_format == "jpg" else []
    with index_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "frame_index", "timestamp_ms", "timestamp_source"])
        for rec in iter_frames(path, start_frame=start_frame, end_frame=end_frame, step=step):
            name = f"{stem}_f{rec.index:06d}.{image_format}"
            if not cv2.imwrite(str(out / name), rec.image, params):
                raise OSError(f"Failed to write {name}")
            writer.writerow([name, rec.index, f"{rec.timestamp_ms:.3f}", rec.timestamp_source.value])
    return index_path


def write_video(frames: Iterator[np.ndarray], out_path: str | os.PathLike, fps: float | Fraction) -> Path:
    """Encode BGR frames to a browser-playable H.264 MP4 (falls back to MPEG-4 Part 2)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rate = Fraction(fps).limit_denominator(1001)
    container = av.open(str(out), mode="w")
    stream = None
    try:
        for img in frames:
            if stream is None:
                h, w = img.shape[:2]
                for codec in ("libx264", "h264", "libopenh264", "mpeg4"):
                    try:
                        stream = container.add_stream(codec, rate=rate)
                        break
                    except (av.error.FFmpegError, ValueError, av.codec.codec.UnknownCodecError):
                        continue
                if stream is None:
                    raise RuntimeError("No usable video encoder available")
                stream.width, stream.height = w - (w % 2), h - (h % 2)
                stream.pix_fmt = "yuv420p"
                if stream.codec_context.name == "libx264":
                    stream.options = {"crf": "20", "preset": "veryfast", "movflags": "+faststart"}
            frame = av.VideoFrame.from_ndarray(img[: stream.height, : stream.width], format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()
    return out
