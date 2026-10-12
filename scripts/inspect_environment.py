"""Report hardware and software relevant to this project (writes outputs/reports/environment.json)."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        return subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:
        return f"error: {exc}"


def main() -> int:
    info: dict = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                  "machine": platform.machine(), "processor": platform.processor()}
    info["nvidia_smi"] = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    info["node"] = run(["node", "--version"])
    info["npm"] = run(["npm", "--version"])
    info["git"] = run(["git", "--version"])
    info["docker"] = run(["docker", "--version"])
    info["ffmpeg_cli"] = run(["ffmpeg", "-version"])
    if info["ffmpeg_cli"]:
        info["ffmpeg_cli"] = info["ffmpeg_cli"].splitlines()[0]
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_gb"] = {"total": round(vm.total / 1e9, 1), "available": round(vm.available / 1e9, 1)}
    except ImportError:
        pass
    du = shutil.disk_usage(ROOT)
    info["disk_gb"] = {"total": round(du.total / 1e9), "free": round(du.free / 1e9)}
    libs = {}
    for mod in ("numpy", "scipy", "cv2", "av", "fastapi", "pydantic", "sqlalchemy", "ultralytics", "torch"):
        try:
            m = importlib.import_module(mod)
            libs[mod] = getattr(m, "__version__", "present")
        except Exception as exc:  # ImportError or OSError (e.g. DLL blocked by Smart App Control)
            libs[mod] = f"unavailable: {type(exc).__name__}: {str(exc)[:160]}"
    info["libraries"] = libs
    try:
        import torch

        info["torch_cuda"] = {"available": torch.cuda.is_available(), "build": torch.version.cuda,
                              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as exc:
        info["torch_cuda"] = f"unavailable: {exc}"[:200]
    out = ROOT / "outputs" / "reports" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
