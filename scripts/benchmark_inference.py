"""Measure detector throughput, latency and GPU memory for several batch sizes / image sizes.

    python scripts/benchmark_inference.py VIDEO --weights models/checkpoints/yolo11s.pt --frames 200
    python scripts/benchmark_inference.py VIDEO --weights models/checkpoints/cricket_ball_best.pt \
        --imgsz 1280 --engine --e2e 600

Writes outputs/reports/inference_benchmark_<host>.json (or --out). Decoding is measured separately
so detector speed and end-to-end speed are not conflated.

* ``--engine`` also runs the validated TensorRT engine next to the weights (see export_tensorrt.py).
* ``--e2e N`` measures detection over the first N frames of the clip, decoding included, twice:
  serially (decode a batch, then detect it) and overlapped (a decoder thread fills a bounded queue
  while the GPU works). This is detection only, without tracking or rendering.

GPU memory: ``gpu_peak_mb`` is PyTorch's allocator peak, which does not see TensorRT's own
allocations, so ``device_mem_delta_mb`` (device-level used memory before vs after the run, from
cudaMemGetInfo) is reported for every run as the comparable figure.
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.detection.base import FrameInfo  # noqa: E402
from ml.video.ingest import iter_frames, probe_video  # noqa: E402


def _used_mb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e6


def bench_detector(det, frames, bs: int) -> dict:
    import torch

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
    base = _used_mb()
    det.detect_batch([frames[0][0]] * bs, [frames[0][1]] * bs)  # warm-up (and engine context creation)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    lat = []
    t1 = time.perf_counter()
    for i in range(0, len(frames), bs):
        chunk = frames[i : i + bs]
        s = time.perf_counter()
        det.detect_batch([c[0] for c in chunk], [c[1] for c in chunk])
        if cuda:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - s) * 1000)
    el = time.perf_counter() - t1
    used = _used_mb()
    lat.sort()
    return {"fps": len(frames) / el, "batch_latency_ms_median": lat[len(lat) // 2],
            "batch_latency_ms_p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1e6 if cuda else None,
            "device_mem_delta_mb": (used - base) if (used is not None and base is not None) else None,
            "effective_batch": det.batch_size}


def e2e(video: str, det, n: int, bs: int, overlap: bool, qsize: int = 4) -> dict:
    """Decode + detect the first n frames; serial or with a decoder thread and a bounded queue."""
    import torch

    def batches():
        buf = []
        for r in iter_frames(video, max_frames=n):
            buf.append((r.image, FrameInfo(r.index, r.timestamp_ms, r.timestamp_source)))
            if len(buf) == bs:
                yield buf
                buf = []
        if buf:
            yield buf

    det.detect_batch([next(iter_frames(video, max_frames=1)).image] * bs,
                     [FrameInfo(0, 0.0, "container_pts")] * bs)  # warm-up
    torch.cuda.synchronize()
    count, wait_s = 0, 0.0
    t0 = time.perf_counter()
    if not overlap:
        for b in batches():
            det.detect_batch([x[0] for x in b], [x[1] for x in b])
            count += len(b)
    else:
        q: queue.Queue = queue.Queue(maxsize=qsize)
        err: list[BaseException] = []

        def producer():
            try:
                for b in batches():
                    q.put(b)
            except BaseException as exc:  # surfaced to the consumer
                err.append(exc)
            finally:
                q.put(None)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        while True:
            w = time.perf_counter()
            b = q.get()
            wait_s += time.perf_counter() - w
            if b is None:
                break
            det.detect_batch([x[0] for x in b], [x[1] for x in b])
            count += len(b)
        th.join()
        if err:
            raise err[0]
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    return {"mode": "overlapped" if overlap else "serial", "frames": count, "fps": count / el,
            "consumer_wait_s": round(wait_s, 3) if overlap else None, "queue_batches": qsize if overlap else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default=str(ROOT / "models/checkpoints/yolo11s.pt"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--batches", default="1,4,8,16")
    ap.add_argument("--imgsz", default="640,1280")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16-only", action="store_true", help="skip the FP32 PyTorch runs")
    ap.add_argument("--engine", action="store_true", help="also benchmark the validated TensorRT engine")
    ap.add_argument("--e2e", type=int, default=0, help="end-to-end decode+detect over this many frames")
    ap.add_argument("--e2e-batch", type=int, default=8)
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
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "weights": Path(a.weights).name, "runs": [], "e2e": []}
    try:
        import tensorrt

        results["tensorrt"] = tensorrt.__version__
    except ImportError:
        results["tensorrt"] = None

    def detectors(imgsz: int, bs: int):
        cuda = torch.cuda.is_available()
        for half in ([True] if a.fp16_only else [True, False]) if cuda else [False]:
            yield "pytorch", half, YoloBallDetector(a.weights, imgsz=imgsz, batch_size=bs, half=half, device=a.device)
        if a.engine and cuda:
            d = YoloBallDetector(a.weights, imgsz=imgsz, batch_size=bs, device=a.device, prefer_engine=True)
            if d.uses_engine:
                yield "tensorrt", True, d
            else:
                print(f"TensorRT engine not used: {d.engine_note}")

    for imgsz in map(int, a.imgsz.split(",")):
        for bs in map(int, a.batches.split(",")):
            for runtime, half, det in detectors(imgsz, bs):
                run = {"runtime": runtime, "imgsz": imgsz, "batch": bs, "half": half, "device": det.device,
                       **bench_detector(det, frames, bs)}
                results["runs"].append(run)
                print(json.dumps(run))
                del det
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if a.e2e and torch.cuda.is_available():
        imgsz = int(a.imgsz.split(",")[-1])
        kinds = [("pytorch", False)] + ([("tensorrt", True)] if a.engine else [])
        for runtime, eng in kinds:
            det = YoloBallDetector(a.weights, imgsz=imgsz, batch_size=a.e2e_batch, half=True, device=a.device,
                                   prefer_engine=eng)
            if eng and not det.uses_engine:
                continue
            for overlap in (False, True, False, True):  # alternate to spread thermal/clock drift
                r = {"runtime": runtime, "imgsz": imgsz, "batch": a.e2e_batch, **e2e(a.video, det, a.e2e, a.e2e_batch, overlap)}
                results["e2e"].append(r)
                print(json.dumps(r))
            del det
            torch.cuda.empty_cache()

    out = Path(a.out or ROOT / f"outputs/reports/inference_benchmark_{platform.node()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"decode fps {results['decode_fps']:.1f}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
