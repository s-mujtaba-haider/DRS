"""Ultralytics YOLO ball detector with optional tiling and OOM recovery.

Two kinds of weights are supported:
* a cricket-specific model trained by ``scripts/train_detector.py`` (class named "ball"/"cricket_ball"), or
* a generic COCO model, using the "sports ball" class. This is NOT cricket specific
  and is reported as such (``is_cricket_specific=False``).

TensorRT: when ``prefer_engine`` is set and ``<weights>.engine`` sits next to the ``.pt`` together
with a validation sidecar ``<weights>.engine.json`` written by ``scripts/export_tensorrt.py``, the
engine is used instead. The sidecar must match the ``.pt`` hash, the installed TensorRT version and
the GPU, and record that the engine's detections agreed with the ``.pt`` model; otherwise the
``.pt`` is used and the reason is logged and exposed as ``engine_note``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np

from ml.common.schemas import Detection
from ml.detection.base import FrameInfo, ModelUnavailableError, make_detection, nms, tile_grid

log = logging.getLogger(__name__)

BALL_CLASS_NAMES = ("ball", "cricket_ball", "cricket ball", "cricket-ball", "sports ball")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def validated_engine(weights: Path) -> tuple[Path | None, str]:
    """Return (engine path, note) if a validated TensorRT engine exists for these .pt weights."""
    eng, side = weights.with_suffix(".engine"), weights.with_suffix(".engine.json")
    if weights.suffix != ".pt" or not eng.is_file():
        return None, "no TensorRT engine"
    if not side.is_file():
        return None, f"{eng.name} has no validation sidecar; using {weights.name}"
    try:
        meta = json.loads(side.read_text())
        import tensorrt
        import torch

        problems = []
        if meta.get("source_sha256") != file_sha256(weights):
            problems.append("engine was built from different .pt weights")
        if meta.get("engine_sha256") != file_sha256(eng):
            problems.append("engine file differs from the one that was validated")
        if meta.get("tensorrt") != tensorrt.__version__:
            problems.append(f"built with TensorRT {meta.get('tensorrt')}, installed {tensorrt.__version__}")
        if not torch.cuda.is_available() or meta.get("gpu") != torch.cuda.get_device_name(0):
            problems.append(f"built for GPU {meta.get('gpu')}")
        if not (meta.get("validation") or {}).get("passed"):
            problems.append("engine did not pass detection-agreement validation")
    except Exception as exc:  # missing tensorrt/torch or unreadable sidecar
        return None, f"TensorRT engine unusable ({type(exc).__name__}: {exc}); using {weights.name}"
    if problems:
        return None, f"TensorRT engine not used: {'; '.join(problems)}"
    return eng, f"TensorRT FP16 engine {eng.name} (validated against {weights.name})"


class YoloBallDetector:
    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.15,
        imgsz: int = 1280,
        device: str = "auto",
        half: bool = True,
        batch_size: int = 8,
        tile_size: int | None = None,
        tile_overlap: float = 0.2,
        max_det_per_frame: int = 10,
        allow_download: bool = False,
        prefer_engine: bool = False,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ModelUnavailableError("ultralytics is not installed") from exc
        except OSError as exc:  # e.g. Windows Smart App Control blocking unsigned torch DLLs
            raise ModelUnavailableError(
                f"PyTorch could not be loaded ({exc}). On Windows with Smart App Control enforced, run the "
                "GPU worker in Docker (see docs/INSTALLATION.md)."
            ) from exc
        wpath = Path(weights)
        if not wpath.is_file() and not allow_download:
            raise ModelUnavailableError(
                f"Detector weights not found: {wpath}. Run scripts/download_models.py or train a model."
            )
        self.engine_note: str | None = None
        self.uses_engine = False
        self._pt_path: Path | None = None
        self._pt_model = None  # lazily loaded fallback for frames outside the engine's input profile
        self._engine_max_hw: tuple[int, int] | None = None
        source_name = wpath.name
        if prefer_engine and wpath.is_file():
            eng, self.engine_note = validated_engine(wpath)
            if eng is not None:
                meta = json.loads(eng.with_suffix(".engine.json").read_text())
                if int(meta.get("imgsz") or imgsz) != imgsz or tile_size:
                    self.engine_note = (f"TensorRT engine not used: built for imgsz {meta.get('imgsz')} without tiling, "
                                        f"requested imgsz {imgsz}{' with tiling' if tile_size else ''}")
                    log.warning(self.engine_note)
                else:
                    self._pt_path, self.uses_engine, wpath = wpath, True, eng
                    batch_size = min(batch_size, int(meta.get("max_batch") or batch_size))  # build-time batch ceiling
                    self._engine_max_hw = tuple(meta.get("max_hw") or (imgsz, imgsz))
            elif self.engine_note != "no TensorRT engine":
                log.warning(self.engine_note)
        self.model = YOLO(str(wpath), task="detect")
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        ball_ids = [k for k, v in names.items() if v in BALL_CLASS_NAMES]
        if not ball_ids:
            raise ModelUnavailableError(f"Model {wpath.name} has no ball class (classes: {sorted(names.values())})")
        self.class_ids = ball_ids
        self.is_cricket_specific = not all(names[c] == "sports ball" for c in ball_ids)
        # The model id names the source weights; the TensorRT engine is a faster runtime of the same model.
        self.model_id = f"yolo:{source_name}" + ("" if self.is_cricket_specific else ":coco-sports-ball")
        if self.uses_engine:
            self.model_id += ":trt-fp16"
        self.conf, self.imgsz, self.max_det = conf, imgsz, max_det_per_frame
        self.tile_size, self.tile_overlap = tile_size, tile_overlap
        self.batch_size = max(1, batch_size)
        self.device = self._resolve_device(device)
        if self.uses_engine and self.device == "cpu":
            raise ModelUnavailableError("A TensorRT engine needs a CUDA device")
        self.half = half and self.device != "cpu"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _precision_kwargs(self) -> dict:
        # Newer Ultralytics replaces the deprecated `half` flag with `quantize` (16 = FP16).
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
        except ImportError:  # pragma: no cover
            DEFAULT_CFG_DICT = {}
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16 if self.half else 32}
        return {"half": self.half}

    def _fits_engine(self, img: np.ndarray, imgsz: int) -> bool:
        """Does this frame's letterboxed shape (Ultralytics rect mode, stride 32) fit the engine profile?"""
        h, w = img.shape[:2]
        r = imgsz / max(h, w)
        lh, lw = (math.ceil(round(v * r) / 32) * 32 for v in (h, w))
        return lh <= self._engine_max_hw[0] and lw <= self._engine_max_hw[1]

    def _predict(self, images: list[np.ndarray], imgsz: int):
        if self.uses_engine and images and not all(self._fits_engine(im, imgsz) for im in images):
            # Ultralytics would silently resize such frames to fit the profile, a different (unvalidated)
            # input than the .pt model sees, so they go to the .pt model instead.
            if self._pt_model is None:
                from ultralytics import YOLO

                log.warning("Frame shape %s is outside the TensorRT profile %s; using %s for it",
                            images[0].shape[:2], self._engine_max_hw, self._pt_path.name)
                self._pt_model = YOLO(str(self._pt_path), task="detect")
            out, keep = [], self.model
            try:
                self.model, self.uses_engine = self._pt_model, False
                out = self._predict(images, imgsz)
            finally:
                self.model, self.uses_engine = keep, True
            return out
        import torch

        while True:
            try:
                out = []
                for i in range(0, len(images), self.batch_size):
                    out.extend(
                        self.model.predict(
                            images[i : i + self.batch_size],
                            conf=self.conf,
                            imgsz=imgsz,
                            device=self.device,
                            **self._precision_kwargs(),
                            classes=self.class_ids,
                            max_det=self.max_det * 4,
                            verbose=False,
                        )
                    )
                return out
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size > 1:
                    self.batch_size //= 2
                    log.warning("CUDA OOM: reducing detector batch size to %d", self.batch_size)
                elif self.device != "cpu" and not self.uses_engine:
                    log.warning("CUDA OOM at batch size 1: falling back to CPU inference")
                    self.device, self.half = "cpu", False
                else:
                    raise

    def detect_batch(self, images: list[np.ndarray], infos: list[FrameInfo]) -> list[list[Detection]]:
        if not images:
            return []
        if not self.tile_size:
            results = self._predict(images, self.imgsz)
            return [self._to_detections(r.boxes, info, 0, 0) for r, info in zip(results, infos)]

        # Tiled inference: each tile is processed at native resolution so tiny balls keep their pixels.
        out: list[list[Detection]] = []
        for img, info in zip(images, infos):
            h, w = img.shape[:2]
            windows = tile_grid(w, h, self.tile_size, self.tile_overlap)
            crops = [img[y0:y1, x0:x1] for (x0, y0, x1, y1) in windows]
            results = self._predict(crops, self.tile_size)
            boxes, scores = [], []
            for r, (x0, y0, _, _) in zip(results, windows):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                b = r.boxes.xyxy.cpu().numpy()
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                boxes.append(b)
                scores.append(r.boxes.conf.cpu().numpy())
            if not boxes:
                out.append([])
                continue
            B, S = np.concatenate(boxes), np.concatenate(scores)
            keep = nms(B, S, 0.4)[: self.max_det]
            out.append([make_detection(info, B[k], S[k], self.model_id, "model") for k in keep])
        return out

    def _to_detections(self, boxes, info: FrameInfo, dx: float, dy: float) -> list[Detection]:
        if boxes is None or len(boxes) == 0:
            return []
        b = boxes.xyxy.cpu().numpy()
        s = boxes.conf.cpu().numpy()
        order = np.argsort(-s)[: self.max_det]
        return [make_detection(info, b[i] + [dx, dy, dx, dy], s[i], self.model_id, "model") for i in order]
