"""Ultralytics YOLO ball detector with optional tiling and OOM recovery.

Two kinds of weights are supported:
* a cricket-specific model trained by ``scripts/train_detector.py`` (class named "ball"/"cricket_ball"), or
* a generic COCO model, using the "sports ball" class. This is NOT cricket specific
  and is reported as such (``is_cricket_specific=False``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ml.common.schemas import Detection
from ml.detection.base import FrameInfo, ModelUnavailableError, make_detection, nms, tile_grid

log = logging.getLogger(__name__)

BALL_CLASS_NAMES = ("ball", "cricket_ball", "cricket ball", "cricket-ball", "sports ball")


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
        self.model = YOLO(str(wpath))
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        ball_ids = [k for k, v in names.items() if v in BALL_CLASS_NAMES]
        if not ball_ids:
            raise ModelUnavailableError(f"Model {wpath.name} has no ball class (classes: {sorted(names.values())})")
        self.class_ids = ball_ids
        self.is_cricket_specific = not all(names[c] == "sports ball" for c in ball_ids)
        self.model_id = f"yolo:{wpath.name}" + ("" if self.is_cricket_specific else ":coco-sports-ball")
        self.conf, self.imgsz, self.max_det = conf, imgsz, max_det_per_frame
        self.tile_size, self.tile_overlap = tile_size, tile_overlap
        self.batch_size = max(1, batch_size)
        self.device = self._resolve_device(device)
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

    def _predict(self, images: list[np.ndarray], imgsz: int):
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
                elif self.device != "cpu":
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
