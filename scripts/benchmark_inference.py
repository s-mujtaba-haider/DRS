"""Measure detector throughput, latency and GPU memory for several batch sizes / image sizes.

    python scripts/benchmark_inference.py VIDEO --weights models/checkpoints/yolo11s.pt --frames 200

Writes outputs/reports/inference_benchmark_<host>.json. Decoding is measured separately
so detector speed and end-to-end speed are not conflated.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.detection.base import FrameInfo  # noqa: E402
from ml.video.ingest import iter_frames, probe_video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default=str(ROOT / "models/checkpoints/yolo11s.pt"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--batches", default="1,4,8,16")
    ap.add_argument("--imgsz", default="640,1280")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch

    from ml.detection.yolo_detector import YoloBallDetector

    meta = probe_video(a.video)
    t0 = time.perf_counter()
    frames = [(r.image, FrameInfo(r.index, r.timestamp_ms, r.timestamp_source)) for r in iter_frames(a.video, max_frames=a.frames)]
    decode_s = time.perf_counter() - t0
    results = {"video": Path(a.video).name, "resolution": [meta.width, meta.height], "frames": len(frames),
               "decode_fps": len(frames) / decode_s, "host": platform.node(), "platform": platform.platform(),
               "torch": torch.__version__, "cuda": torch.cuda.is_available(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "runs": []}
    for imgsz in map(int, a.imgsz.split(",")):
        for bs in map(int, a.batches.split(",")):
            for half in ([True, False] if torch.cuda.is_available() else [False]):
                det = YoloBallDetector(a.weights, imgsz=imgsz, batch_size=bs, half=half, device=a.device)
                det.detect_batch([frames[0][0]] * bs, [frames[0][1]] * bs)  # warm-up
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                lat = []
                t1 = time.perf_counter()
                for i in range(0, len(frames), bs):
                    chunk = frames[i : i + bs]
                    s = time.perf_counter()
                    det.detect_batch([c[0] for c in chunk], [c[1] for c in chunk])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    lat.append((time.perf_counter() - s) * 1000)
                el = time.perf_counter() - t1
                run = {"imgsz": imgsz, "batch": bs, "half": half, "device": det.device, "fps": len(frames) / el,
                       "batch_latency_ms_median": sorted(lat)[len(lat) // 2],
                       "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None,
                       "effective_batch": det.batch_size}
                results["runs"].append(run)
                print(json.dumps(run))
                del det
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    out = Path(a.out or ROOT / f"outputs/reports/inference_benchmark_{platform.node()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"decode fps {results['decode_fps']:.1f}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
