"""Model *proposals* of batter ground-contact points for run-out / stumping review.

A generic COCO human-pose model (YOLO11s-pose) finds people; the person nearest the crease
(or inside the operator's region) is taken as the batter and its ankle keypoints are proposed
as candidate ground-contact points. Everything returned here is a suggestion for the operator
to confirm or correct in the crease workspace. Nothing is fed to the crease engine as an
observation until the operator accepts it (it then becomes a ``source="model"`` observation).

Not cricket-trained:
* the pose model was trained on COCO people, not batters in pads; ankles under pads and
  shoes in motion blur are often mislocated by several pixels;
* there is no bat model, so bat-tip proposals are not produced (the bat is usually what
  is grounded first in a run-out);
* ``toe_est`` extends the shin (knee -> ankle) by a fixed fraction to approximate the sole,
  which is a geometric guess, not a detection;
* ``grounded_hint`` is a heuristic from vertical image motion of the foot relative to the
  person's height. Camera motion, foreshortening and a foot sliding along the ground all
  break it. It is a hint for the operator, never an assertion.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

PROPOSALS_VERSION = "crease-proposals-0.1"
POSE_LICENCE = "AGPL-3.0 (Ultralytics YOLO11 pose weights, trained on COCO keypoints)"
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "yolo11s-pose.pt"
MAX_FRAMES = 300

# COCO keypoint indices
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16
L_WRIST, R_WRIST = 9, 10

ProposalKind = Literal["ankle_left", "ankle_right", "toe_est", "bat_tip"]
NOTE = ("Model proposal (generic COCO pose model, not cricket-trained). Confirm or correct on the frame "
        "before it is used as evidence.")


@dataclass
class PersonPose:
    box: tuple[float, float, float, float]
    score: float
    keypoints: np.ndarray  # (17, 3): x, y, confidence

    @property
    def height(self) -> float:
        return max(1.0, self.box[3] - self.box[1])

    def foot_point(self) -> tuple[float, float]:
        """Ankle midpoint when visible, else the bottom-centre of the box."""
        k = self.keypoints
        vis = [i for i in (L_ANKLE, R_ANKLE) if k[i, 2] >= 0.3]
        if vis:
            return float(np.mean(k[vis, 0])), float(np.mean(k[vis, 1]))
        return (self.box[0] + self.box[2]) / 2, self.box[3]


class PoseModel(Protocol):
    model_id: str

    def predict(self, images: list[np.ndarray]) -> list[list[PersonPose]]: ...


class PoseUnavailableError(RuntimeError):
    pass


class UltralyticsPose:
    """YOLO pose wrapper: GPU FP16 when available, batch halving and CPU fallback on CUDA OOM."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, imgsz: int = 1280, conf: float = 0.25,
                 device: str = "auto", batch_size: int = 8):
        wpath = Path(weights)
        if not wpath.is_file():
            raise PoseUnavailableError(f"Pose weights not found: {wpath}. Download with "
                                       "`python -c \"from ultralytics import YOLO; YOLO('models/checkpoints/yolo11s-pose.pt')\"`")
        try:
            import torch
            from ultralytics import YOLO
        except (ImportError, OSError) as exc:
            raise PoseUnavailableError(f"ultralytics/PyTorch unavailable: {exc}") from exc
        self.model = YOLO(str(wpath))
        if getattr(self.model, "task", None) != "pose":
            raise PoseUnavailableError(f"{wpath.name} is not a pose model")
        self.device = ("0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.half = self.device != "cpu"
        self.imgsz, self.conf, self.batch_size = imgsz, conf, max(1, batch_size)
        self.model_id = f"yolo-pose:{wpath.name}@{imgsz}"

    def _precision(self) -> dict:
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
        except ImportError:  # pragma: no cover
            DEFAULT_CFG_DICT = {}
        return {"quantize": 16 if self.half else 32} if "quantize" in DEFAULT_CFG_DICT else {"half": self.half}

    def predict(self, images: list[np.ndarray]) -> list[list[PersonPose]]:
        import torch

        while True:
            try:
                out: list[list[PersonPose]] = []
                for i in range(0, len(images), self.batch_size):
                    res = self.model.predict(images[i:i + self.batch_size], imgsz=self.imgsz, conf=self.conf,
                                             device=self.device, verbose=False, **self._precision())
                    for r in res:
                        people = []
                        if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
                            b = r.boxes.xyxy.cpu().numpy()
                            s = r.boxes.conf.cpu().numpy()
                            kp = r.keypoints.data.cpu().numpy()  # (n, 17, 3)
                            for j in range(len(b)):
                                people.append(PersonPose(tuple(float(v) for v in b[j]), float(s[j]), kp[j].astype(float)))
                        out.append(people)
                return out
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size > 1:
                    self.batch_size //= 2
                    log.warning("CUDA OOM: pose batch size -> %d", self.batch_size)
                elif self.device != "cpu":
                    log.warning("CUDA OOM at batch 1: pose falls back to CPU")
                    self.device, self.half = "cpu", False
                else:
                    raise


# ----------------------------------------------------------------- selection
def _dist_point_line(p, a, b) -> float:
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-9:
        return float(np.linalg.norm(p - a))
    return float(abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / n)


def _in_roi(p, roi) -> bool:
    return roi[0] <= p[0] <= roi[2] and roi[1] <= p[1] <= roi[3]


def select_batter(people: list[PersonPose], crease: tuple | None = None, roi: tuple | None = None,
                  previous: tuple[float, float] | None = None, image_size: tuple[int, int] | None = None
                  ) -> tuple[PersonPose | None, str]:
    """Pick the person most likely to be the batter. Returns (person, reason).

    Score (lower is better), all in units of the person's own height so near and far people compare:
    distance of the feet to the crease line, a large penalty outside the ROI, and distance to the
    previous frame's selection (temporal continuity).
    """
    if crease is None and roi is None:
        # Measured on broadcast clips: "largest person" picked the umpire, non-striker or bowler in 7 of 16.
        return None, "crease line or region required to identify the batter"
    if not people:
        return None, "no person detected"
    best, best_score = None, math.inf
    for p in people:
        foot = p.foot_point()
        h = p.height
        score = 0.0
        if crease is not None:
            score += _dist_point_line(foot, crease[0], crease[1]) / h
        if roi is not None:
            if not _in_roi(foot, roi):
                score += 10.0
        if previous is not None:
            score += 0.5 * float(np.hypot(foot[0] - previous[0], foot[1] - previous[1])) / h
        score -= 0.1 * p.score
        if score < best_score:
            best, best_score = p, score
    if roi is not None and best is not None and not _in_roi(best.foot_point(), roi):
        return None, "no person with feet inside the region"
    reason = "nearest the crease" if crease is not None else "inside the region"
    return best, reason


# ----------------------------------------------------------------- proposals
class Proposal(BaseModel):
    frame_index: int
    timestamp_ms: float
    point: tuple[float, float]
    kind: ProposalKind
    confidence: float = Field(ge=0, le=1)
    grounded_hint: bool | None = None
    hint_basis: str | None = None
    source: Literal["model"] = "model"
    model_id: str
    note: str = NOTE


class FrameProposals(BaseModel):
    frame_index: int
    timestamp_ms: float
    batter_found: bool
    selection: str
    person_box: tuple[float, float, float, float] | None = None
    person_score: float | None = None
    n_people: int = 0
    proposals: list[Proposal] = Field(default_factory=list)


class ProposalsResult(BaseModel):
    version: str = PROPOSALS_VERSION
    model_id: str
    licence: str = POSE_LICENCE
    frames: list[FrameProposals]
    frames_with_batter: int
    inference_fps: float | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = ("Proposals from a generic human-pose model. They are suggestions for the operator to confirm, "
                       "not observations. Bat tips are not proposed.")


@dataclass
class HintConfig:
    still_frac: float = 0.012  # |vertical velocity| / person height per frame below this -> "still"
    moving_frac: float = 0.05  # above this -> clearly moving vertically
    lower_foot_frac: float = 0.03  # a foot within this of the lower foot counts as "lower"
    min_kp_conf: float = 0.5
    toe_extension: float = 0.15  # shin fraction added below the ankle for the sole estimate


def _grounded_hints(seq: list[tuple[PersonPose | None]], cfg: HintConfig) -> list[dict[int, tuple[bool | None, str]]]:
    """Per frame, per ankle index: (hint, basis). Uses the frames on both sides."""
    out: list[dict[int, tuple[bool | None, str]]] = []
    for t, p in enumerate(seq):
        hints: dict[int, tuple[bool | None, str]] = {}
        if p is None:
            out.append(hints)
            continue
        h = p.height
        for a, other in ((L_ANKLE, R_ANKLE), (R_ANKLE, L_ANKLE)):
            k = p.keypoints
            if k[a, 2] < cfg.min_kp_conf:
                hints[a] = (None, "low keypoint confidence")
                continue
            nb = [seq[t - 1] if t > 0 else None, seq[t + 1] if t + 1 < len(seq) else None]
            if any(q is None or q.keypoints[a, 2] < cfg.min_kp_conf for q in nb):
                hints[a] = (None, "foot not visible in both neighbouring frames")
                continue
            vy = (nb[1].keypoints[a, 1] - nb[0].keypoints[a, 1]) / 2.0 / h
            lower = k[other, 2] < cfg.min_kp_conf or k[a, 1] >= k[other, 1] - cfg.lower_foot_frac * h
            if abs(vy) <= cfg.still_frac and lower:
                hints[a] = (True, f"lower foot, vertical motion {vy:+.3f} body-heights/frame")
            elif abs(vy) >= cfg.moving_frac or (not lower and k[other, 1] - k[a, 1] > 0.1 * h):
                hints[a] = (False, f"foot moving or raised (vertical motion {vy:+.3f} body-heights/frame)")
            else:
                hints[a] = (None, f"ambiguous (vertical motion {vy:+.3f} body-heights/frame)")
        out.append(hints)
    return out


def propose_from_poses(frames: list[tuple[int, float]], poses: list[list[PersonPose]], model_id: str,
                       crease: tuple | None = None, roi: tuple | None = None,
                       image_size: tuple[int, int] | None = None, cfg: HintConfig | None = None) -> ProposalsResult:
    """Turn per-frame pose detections into ground-contact proposals for the batter."""
    cfg = cfg or HintConfig()
    warnings: list[str] = []
    if crease is None and roi is None:
        warnings.append("No crease line or region given: the batter cannot be identified, so nothing is proposed")
    warnings.append("Near the stumps the wicket-keeper can be selected instead of the batter; check the person box")
    chosen: list[PersonPose | None] = []
    reasons: list[str] = []
    prev = None
    for people in poses:
        p, why = select_batter(people, crease, roi, prev, image_size)
        chosen.append(p)
        reasons.append(why)
        prev = p.foot_point() if p is not None else prev
    hints = _grounded_hints(chosen, cfg)
    out: list[FrameProposals] = []
    for (fi, ts), people, p, why, hh in zip(frames, poses, chosen, reasons, hints):
        fp = FrameProposals(frame_index=fi, timestamp_ms=ts, batter_found=p is not None, selection=why, n_people=len(people))
        if p is not None:
            fp.person_box, fp.person_score = p.box, round(p.score, 4)
            k = p.keypoints
            for a, knee, kind in ((L_ANKLE, L_KNEE, "ankle_left"), (R_ANKLE, R_KNEE, "ankle_right")):
                if k[a, 2] < 0.3:
                    continue
                g, basis = hh.get(a, (None, None))
                fp.proposals.append(Proposal(frame_index=fi, timestamp_ms=ts, point=(round(k[a, 0], 2), round(k[a, 1], 2)),
                                             kind=kind, confidence=round(float(k[a, 2]), 4), grounded_hint=g, hint_basis=basis,
                                             model_id=model_id))
                if k[knee, 2] >= 0.3:
                    toe = k[a, :2] + cfg.toe_extension * (k[a, :2] - k[knee, :2])
                    fp.proposals.append(Proposal(frame_index=fi, timestamp_ms=ts, point=(round(toe[0], 2), round(toe[1], 2)),
                                                 kind="toe_est", confidence=round(float(min(k[a, 2], k[knee, 2])), 4),
                                                 grounded_hint=g, hint_basis=(basis or "") + "; sole estimated from the shin direction",
                                                 model_id=model_id))
        out.append(fp)
    found = sum(f.batter_found for f in out)
    if out and found < len(out):
        warnings.append(f"Batter not found in {len(out) - found} of {len(out)} frames")
    warnings.append("Bat tip is not proposed (no bat model); mark the bat manually")
    warnings.append("Grounded hints assume a static camera; pans and zooms make them unreliable")
    return ProposalsResult(model_id=model_id, frames=out, frames_with_batter=found, warnings=warnings)


def propose_in_video(path: str | Path, model: PoseModel, start_frame: int, end_frame: int,
                     crease: tuple | None = None, roi: tuple | None = None, batch: int = 16,
                     max_frames: int = MAX_FRAMES) -> ProposalsResult:
    """Run the pose model over [start_frame, end_frame) of a video and build proposals."""
    import time

    from ml.video.ingest import iter_frames

    if end_frame <= start_frame:
        raise ValueError("end_frame must exceed start_frame")
    if end_frame - start_frame > max_frames:
        raise ValueError(f"at most {max_frames} frames per request")
    frames: list[tuple[int, float]] = []
    poses: list[list[PersonPose]] = []
    buf, meta = [], []
    size = None
    t_inf = 0.0

    def flush():
        nonlocal t_inf
        t0 = time.perf_counter()
        poses.extend(model.predict(buf))
        t_inf += time.perf_counter() - t0
        frames.extend(meta)
        buf.clear()
        meta.clear()

    for rec in iter_frames(path, start_frame=start_frame, end_frame=end_frame):
        size = (rec.image.shape[1], rec.image.shape[0])
        buf.append(rec.image)
        meta.append((rec.index, rec.timestamp_ms))
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    res = propose_from_poses(frames, poses, model.model_id, crease, roi, size)
    res.inference_fps = round(len(frames) / t_inf, 2) if t_inf > 0 else None
    if not frames:
        res.warnings.insert(0, "No frames decoded in the requested range")
    return res
