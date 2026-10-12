"""End-to-end single-video pipeline: ingest -> detect -> track -> 2D trajectory -> outputs.

Frames are streamed in small batches; nothing holds the whole video in memory.
The annotated video is rendered in a second streaming pass. Progress and
cancellation hooks let the backend run this in a background worker.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ml.common.schemas import Detection, FrameDetections, TrackingResult
from ml.detection.base import FrameInfo, ModelUnavailableError
from ml.detection.motion_detector import HybridDetector, MotionCandidateDetector
from ml.tracking.tracker import BallTracker, TrackerConfig
from ml.trajectory.image_trajectory import fit_image_trajectory, predict
from ml.video.ingest import iter_frames, probe_video, write_video
from ml.video.shots import ShotCutDetector

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CRICKET_WEIGHTS = ROOT / "models" / "checkpoints" / "cricket_ball_best.pt"
COCO_WEIGHTS = ROOT / "models" / "checkpoints" / "yolo11s.pt"
PIPELINE_VERSION = "pipeline-0.1.0"


class CancelledError(RuntimeError):
    pass


@dataclass
class PipelineConfig:
    detector: str = "auto"  # auto | cricket | coco_hybrid | coco | motion
    weights: str | None = None
    conf: float = 0.15
    imgsz: int = 1280
    tile_size: int | None = None
    batch_size: int = 8
    device: str = "auto"
    half: bool = True
    frame_step: int = 1
    start_frame: int = 0
    end_frame: int | None = None
    max_frames: int | None = 3000
    render_video: bool = True
    detect_shot_cuts: bool = True
    tracker: dict = field(default_factory=dict)


def build_detector(cfg: PipelineConfig) -> tuple[object, list[str]]:
    """Choose a detector, reporting exactly what is (and is not) being used."""
    notes: list[str] = []
    from ml.detection.yolo_detector import YoloBallDetector

    def yolo(path: Path):
        return YoloBallDetector(path, conf=cfg.conf, imgsz=cfg.imgsz, device=cfg.device, half=cfg.half,
                                batch_size=cfg.batch_size, tile_size=cfg.tile_size)

    choice = cfg.detector
    if choice == "motion":
        notes.append("Motion-candidate heuristic only (no learned detector)")
        return MotionCandidateDetector(), notes
    weights = Path(cfg.weights) if cfg.weights else None
    if choice in ("auto", "cricket"):
        w = weights or CRICKET_WEIGHTS
        if w.is_file():
            try:
                det = yolo(w)
                if det.is_cricket_specific:
                    notes.append(f"Cricket-specific detector {w.name}")
                    return det, notes
                notes.append(f"{w.name} is not cricket-specific")
            except ModelUnavailableError as exc:
                notes.append(str(exc))
        elif choice == "cricket":
            raise ModelUnavailableError(f"Cricket ball weights not found at {w}. Train one with scripts/train_detector.py")
        else:
            notes.append("No trained cricket-ball weights found; falling back to generic COCO 'sports ball' + motion candidates")
    w = weights if (weights and choice in ("coco", "coco_hybrid")) else COCO_WEIGHTS
    try:
        det = yolo(w)
    except ModelUnavailableError as exc:
        if choice in ("coco", "coco_hybrid"):
            raise
        notes.append(f"{exc}. Using motion-candidate heuristic only")
        return MotionCandidateDetector(), notes
    notes.append(f"Generic COCO detector {w.name} (class 'sports ball'): NOT trained on cricket footage")
    if choice == "coco":
        return det, notes
    return HybridDetector(det, MotionCandidateDetector()), notes


def _gpu_peak_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except ImportError:
        pass
    return None


def run_pipeline(
    video_path: str | Path,
    out_dir: str | Path,
    cfg: PipelineConfig | None = None,
    progress: Callable[[str, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    detector=None,
) -> dict:
    cfg = cfg or PipelineConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda stage, frac: None)
    should_cancel = should_cancel or (lambda: False)
    timings: dict[str, float] = {}
    warnings: list[str] = []

    t0 = time.perf_counter()
    progress("probe", 0.0)
    meta = probe_video(video_path)
    warnings += meta.warnings
    timings["probe_s"] = time.perf_counter() - t0

    notes: list[str] = []
    if detector is None:
        detector, notes = build_detector(cfg)
    total = meta.frame_count_declared or (int(meta.duration_s * meta.nominal_fps) if meta.duration_s and meta.nominal_fps else None)
    if cfg.end_frame is not None:
        total = min(total or cfg.end_frame, cfg.end_frame)
    if cfg.max_frames:
        total = min(total or cfg.max_frames * cfg.frame_step, cfg.max_frames * cfg.frame_step)
    expected = max(1, ((total or 1) - cfg.start_frame) // cfg.frame_step)

    # ---------------------------------------------------------------- detect
    t1 = time.perf_counter()
    frames: list[tuple[int, float]] = []
    frame_dets: list[FrameDetections] = []
    all_dets: list[Detection] = []
    batch_imgs, batch_infos = [], []

    def flush():
        if not batch_imgs:
            return
        results = detector.detect_batch(batch_imgs, batch_infos)
        for info, dets in zip(batch_infos, results):
            frame_dets.append(FrameDetections(frame_index=info.index, timestamp_ms=info.timestamp_ms,
                                              timestamp_source=info.timestamp_source, ball_detected=bool(dets),
                                              detections=dets))
            all_dets.extend(dets)
        batch_imgs.clear()
        batch_infos.clear()

    ts_sources = set()
    cuts = ShotCutDetector() if cfg.detect_shot_cuts else None
    shot_breaks: set[int] = set()
    for rec in iter_frames(video_path, cfg.start_frame, cfg.end_frame, cfg.frame_step, cfg.max_frames):
        if should_cancel():
            raise CancelledError("Cancelled during detection")
        if cuts is not None and cuts.is_cut(rec.image):
            shot_breaks.add(rec.index)
            if hasattr(detector, "motion"):
                detector.motion.reset()  # frame differencing across a cut is meaningless
            elif hasattr(detector, "reset"):
                detector.reset()
        info = FrameInfo(rec.index, rec.timestamp_ms, rec.timestamp_source)
        ts_sources.add(rec.timestamp_source.value)
        frames.append((rec.index, rec.timestamp_ms))
        batch_imgs.append(rec.image)
        batch_infos.append(info)
        if len(batch_imgs) >= cfg.batch_size:
            flush()
            progress("detect", min(0.99, len(frames) / expected))
    flush()
    timings["detect_s"] = time.perf_counter() - t1
    if not frames:
        raise ValueError("No frames decoded from video")
    if "estimated_from_fps" in ts_sources:
        warnings.append("Some timestamps were estimated from the nominal frame rate")

    # ----------------------------------------------------------------- track
    progress("track", 0.0)
    t2 = time.perf_counter()
    tracker = BallTracker(TrackerConfig(**cfg.tracker))
    tracking: TrackingResult = tracker.run(frames, all_dets, shot_breaks)
    timings["track_s"] = time.perf_counter() - t2
    warnings += tracking.warnings

    # ------------------------------------------------------------ trajectory
    traj = None
    fitted_px: list[tuple[float, float]] = []
    primary = next((t for t in tracking.tracks if t.is_primary), None)
    if primary:
        obs = [p for p in primary.points if p.kind == "observed"]
        if obs:
            ts = np.array([p.timestamp_ms / 1000 for p in obs])
            traj = fit_image_trajectory(ts, np.array([p.x for p in obs]), np.array([p.y for p in obs]))
            if traj.segments:
                grid = np.linspace(traj.segments[0].t_start_s, traj.segments[-1].t_end_s, 60)
                pr = predict(traj, grid)
                fitted_px = list(zip(pr["x"].tolist(), pr["y"].tolist()))

    # --------------------------------------------------------------- outputs
    model_id = getattr(detector, "model_id", "unknown")
    (out / "metadata.json").write_text(meta.model_dump_json(indent=2))
    (out / "detections.json").write_text(json.dumps([fd.model_dump(mode="json") for fd in frame_dets]))
    (out / "tracking.json").write_text(tracking.model_dump_json())
    if traj is not None:
        (out / "trajectory_2d.json").write_text(traj.model_dump_json(indent=2))
    with (out / "track_points.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "is_primary", "frame_index", "timestamp_ms", "kind", "x", "y", "sigma_x", "sigma_y", "confidence", "source"])
        for tr in tracking.tracks:
            for p in tr.points:
                w.writerow([tr.track_id, tr.is_primary, p.frame_index, f"{p.timestamp_ms:.3f}", p.kind, f"{p.x:.2f}", f"{p.y:.2f}",
                            "" if p.sigma_x is None else f"{p.sigma_x:.2f}", "" if p.sigma_y is None else f"{p.sigma_y:.2f}",
                            "" if p.confidence is None else f"{p.confidence:.3f}", p.detection_source or ""])

    if cfg.render_video:
        progress("render", 0.0)
        t3 = time.perf_counter()
        from ml.visualization.overlay import OverlayRenderer

        renderer = OverlayRenderer(tracking, all_dets, model_id, fitted_path=fitted_px)

        def gen():
            for i, rec in enumerate(iter_frames(video_path, cfg.start_frame, cfg.end_frame, cfg.frame_step, cfg.max_frames)):
                if should_cancel():
                    raise CancelledError("Cancelled during rendering")
                if i % 25 == 0:
                    progress("render", min(0.99, i / len(frames)))
                yield renderer.draw(rec.image, rec.index, rec.timestamp_ms)

        out_fps = (meta.nominal_fps or 25.0) / cfg.frame_step
        write_video(gen(), out / "annotated.mp4", out_fps)
        timings["render_s"] = time.perf_counter() - t3

    timings["total_s"] = time.perf_counter() - t0
    n = len(frames)
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "video": meta.path_name,
        "config": asdict(cfg),
        "detector": {"model_id": model_id, "cricket_specific": bool(getattr(detector, "is_cricket_specific", False)), "notes": notes},
        "frames_processed": n,
        "frames_with_candidates": tracking.num_frames_with_candidates,
        "num_detections": len(all_dets),
        "num_tracks": len(tracking.tracks),
        "shot_cuts": sorted(shot_breaks),
        "primary_track": primary.quality.model_dump() if primary else None,
        "trajectory_2d": None if traj is None else {"status": traj.status, "model": traj.model_name,
                                                    "rms_px": traj.residual_rms_px, "loo_rms_px": traj.loo_rms_px,
                                                    "change_point_time_s": traj.change_point_time_s},
        "timings": timings,
        "throughput_fps": {"detect": n / timings["detect_s"] if timings["detect_s"] else None,
                           "end_to_end": n / timings["total_s"]},
        "gpu_peak_mb": _gpu_peak_mb(),
        "warnings": sorted(set(warnings)),
        "validation_status": "UNVALIDATED: no ground-truth labels for this video; detections and tracks are not verified",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    progress("done", 1.0)
    return summary
