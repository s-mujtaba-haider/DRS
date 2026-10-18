"""Health, hardware/model status, dataset registry, dashboard statistics and evaluation reports."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from functools import lru_cache

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.api.deps import db
from backend.app.core.config import ROOT
from backend.app.database.models import Job, LbwAnalysis, Video

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@lru_cache(maxsize=1)
def _torch_status() -> dict:
    """Probe PyTorch once. Import failures (e.g. blocked DLLs) are reported, not raised."""
    try:
        import torch

        info = {"available": True, "version": torch.__version__, "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(device=p.name, vram_total_gb=round(p.total_memory / 1e9, 2))
        return info
    except Exception as exc:  # ImportError, OSError (DLL blocked), ...
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:300]}


def _nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        name, driver, total, used, util = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "driver": driver, "memory_total_mb": int(total), "memory_used_mb": int(used),
                "utilization_pct": int(util)}
    except Exception:
        return None


def _models() -> list[dict]:
    ckpt = ROOT / "models" / "checkpoints"
    out = []
    for p in sorted(ckpt.glob("*.pt")) + sorted(ckpt.glob("*.onnx")):
        card = p.with_suffix(".json")
        out.append({"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 1),
                    "card": json.loads(card.read_text()) if card.is_file() else None})
    return out


@router.get("/system/status")
def system_status():
    from ml.pipeline import COCO_WEIGHTS, CRICKET_WEIGHTS

    torch_info = _torch_status()
    detector_mode = (
        "cricket_trained" if torch_info["available"] and CRICKET_WEIGHTS.is_file()
        else "coco_generic+motion" if torch_info["available"] and COCO_WEIGHTS.is_file()
        else "motion_heuristic_only"
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_info,
        "gpu": _nvidia_smi(),
        "models": _models(),
        "cricket_detector_available": CRICKET_WEIGHTS.is_file(),
        "effective_detector_mode": detector_mode,
    }


@router.get("/datasets")
def datasets():
    return json.loads((ROOT / "data" / "dataset_registry.json").read_text())


@router.get("/stats")
def stats(s: Session = Depends(db)):
    jobs_by_status = dict(s.query(Job.status, func.count()).group_by(Job.status).all())
    succeeded = s.query(Job).filter(Job.status == "succeeded").all()
    with_track = sum(1 for j in succeeded if j.summary and j.summary.get("primary_track"))
    return {
        "videos": s.query(func.count(Video.id)).scalar(),
        "jobs_by_status": jobs_by_status,
        "lbw_analyses": s.query(func.count(LbwAnalysis.id)).scalar(),
        # "Track found" is not "track correct": no ground truth exists for uploaded videos.
        "jobs_with_primary_track": with_track,
        "jobs_succeeded": len(succeeded),
        "detection_metrics": _latest_eval("detector_eval"),
        "trajectory_benchmark": _latest_eval("synth_benchmark"),
    }


def _latest_eval(prefix: str) -> dict | None:
    files = sorted((ROOT / "outputs" / "reports").glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    return {"file": files[-1].name, "summary": data.get("summary", data)}


@router.get("/evaluation/reports")
def evaluation_reports():
    d = ROOT / "outputs" / "reports"
    out = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out.append({"file": p.name, "summary": data.get("summary", data) if isinstance(data, dict) else None})
    return out


@router.get("/auth/status")
def auth_status(request: Request):
    """Whether match-day PINs are enabled and which role the supplied X-DRS-PIN maps to."""
    return {"enabled": request.state.auth_enabled, "role": request.state.role or "viewer"}
