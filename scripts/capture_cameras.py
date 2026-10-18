"""Live multi-camera capture into a rolling buffer, with review-clip export.

    python scripts/capture_cameras.py probe                       # list working OpenCV devices
    python scripts/capture_cameras.py record --config configs/capture.yaml --duration 60 --export-last 8
    python scripts/capture_cameras.py benchmark                   # synthetic throughput on this PC

``record`` keeps every configured camera recording until --duration elapses or
Ctrl+C, printing status every few seconds, then optionally exports the last N
seconds of all cameras to outputs/capture/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.capture import CaptureManager, RecorderConfig, SyntheticSource  # noqa: E402
from ml.capture.recorder import ENCODER_PREFERENCE, encoder_works  # noqa: E402


def probe(max_index: int = 5) -> list[dict]:
    import cv2

    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(i)
        ok = cap.isOpened()
        info = {"index": i, "opened": ok}
        if ok:
            got, img = cap.read()
            info.update(read_ok=bool(got), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=cap.get(cv2.CAP_PROP_FPS))
            found.append(info)
        cap.release()
    return found


def cmd_probe(args) -> None:
    print("encoders:", {c: encoder_works(c) for c in ENCODER_PREFERENCE})
    devs = probe(args.max_index)
    print(json.dumps(devs, indent=1) if devs else "No OpenCV capture devices opened")


def cmd_record(args) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())
    rc = RecorderConfig(**cfg.get("recorder", {}))
    mgr = CaptureManager(ROOT / cfg.get("storage_dir", "storage/capture"), rc)
    mgr.start_cameras(cfg["cameras"])
    t_end = time.time() + args.duration if args.duration else None
    try:
        while t_end is None or time.time() < t_end:
            time.sleep(args.status_every)
            for c in mgr.status()["cameras"]:
                print(f"{c['camera_id']:>10}: {c['measured_fps'] or 0:6.1f} fps, captured {c['frames_captured']}, "
                      f"dropped src {c['source_dropped_est']} queue {c['queue_dropped']}, backlog {c['encoder_backlog']}, "
                      f"buffer {c['buffer_span_s']:.0f}s {c['disk_bytes'] / 1e6:.0f} MB {'ERROR ' + c['error'] if c['error'] else ''}")
    except KeyboardInterrupt:
        pass
    try:
        if args.export_last:
            out = ROOT / "outputs" / "capture" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            exp = cfg.get("export", {})
            res = mgr.export_last(args.export_last, out, codec=exp.get("codec", "libx264"), crf=exp.get("crf", 12))
            print(json.dumps({k: v for k, v in res.items() if k != "window"}, indent=1))
    finally:
        mgr.stop_all()


def _bench_one(root: Path, w: int, h: int, fps: float, encoder: str, seconds: float, n_cams: int = 1,
               processes: bool = False, texture: str = "noise") -> dict:
    import psutil

    mgr = CaptureManager(root / f"{encoder}_{w}x{h}_{fps:g}_{n_cams}_{int(processes)}",
                         RecorderConfig(segment_s=2.0, retention_s=30, encoder=encoder), process_per_camera=processes)
    srcs = [{"camera_id": f"cam{i}", "type": "synthetic", "width": w, "height": h, "fps": fps, "texture": texture}
            for i in range(n_cams)]
    mgr.start_cameras(srcs)
    time.sleep(1.0)  # warm-up (process start, first segment)
    procs = [psutil.Process()] + ([psutil.Process(r.proc.pid) for r in mgr.recorders.values()] if processes else [])
    for p in procs:
        p.cpu_percent(None)
    st0 = {c["camera_id"]: c for c in mgr.status()["cameras"]}
    time.sleep(seconds)
    cpu = sum(p.cpu_percent(None) for p in procs)
    st = mgr.status()["cameras"]
    mgr.stop_all()
    for c in st:  # report the measurement window only
        b = st0.get(c["camera_id"], {})
        for k in ("frames_captured", "frames_encoded", "source_dropped_est", "queue_dropped"):
            c[k] = (c.get(k) or 0) - (b.get(k) or 0)
    # synthetic frame generation cost (part of the measured CPU, not of a real camera)
    s = SyntheticSource("gen", w, h, fps, realtime=False)
    s.open()
    t0 = time.perf_counter()
    for _ in range(50):
        s.read()
    gen_ms = (time.perf_counter() - t0) / 50 * 1000
    return {"width": w, "height": h, "nominal_fps": fps, "encoder": st[0]["encoder"], "cameras": n_cams,
            "mode": "process_per_camera" if processes else "threads", "texture": texture,
            "seconds": seconds, "measured_fps": [round(c["measured_fps"] or 0, 2) for c in st],
            "frames_captured": [c["frames_captured"] for c in st], "frames_encoded": [c["frames_encoded"] for c in st],
            "source_dropped_est": [c["source_dropped_est"] for c in st], "queue_dropped": [c["queue_dropped"] for c in st],
            "encode_ms_per_frame": [round(c["encode_ms_per_frame"] or 0, 2) for c in st],
            "arrival_jitter_ms": [round(c["arrival_jitter_ms"] or 0, 3) for c in st],
            "process_cpu_percent_of_one_core": round(cpu, 1), "logical_cpus": psutil.cpu_count(),
            "synthetic_generation_ms_per_frame": round(gen_ms, 2),
            "recorded_mbit_per_s_per_camera": round(sum(c["disk_bytes"] for c in st) * 8 / 1e6 / (seconds + 1.0) / n_cams, 1)}


def cmd_benchmark(args) -> None:
    import tempfile

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for enc in [e for e in ("h264_nvenc", "libx264") if encoder_works(e)]:
            for (w, h) in ((1280, 720), (1920, 1080)):
                for fps in (50.0, 100.0, 120.0):
                    for processes in (False, True):
                        r = _bench_one(Path(td), w, h, fps, enc, args.seconds, processes=processes)
                        rows.append(r)
                        print(json.dumps(r), flush=True)
            for n in (2, 3):
                r = _bench_one(Path(td), 1920, 1080, 100.0, enc, args.seconds, n_cams=n, processes=True)
                rows.append(r)
                print(json.dumps(r), flush=True)
    import platform

    import cv2

    out = ROOT / "outputs" / "reports" / "capture_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"measured_at_utc": datetime.now(UTC).isoformat(), "platform": platform.platform(),
                               "processor": platform.processor(), "opencv": cv2.__version__,
                               "note": "SyntheticSource frames (temporal sensor-like noise, sigma ~4 levels) paced on the "
                                       "host clock; counts cover the measurement window after a 1 s warm-up; CPU is the sum "
                                       "over the parent and camera processes in percent of one logical core and includes "
                                       "synthetic frame generation and BGR->YUV conversion. Real cameras add "
                                       "driver/USB/network cost and real scenes change the bitrate.",
                               "devices_probe": probe(3), "rows": rows}, indent=2))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--max-index", type=int, default=5)
    p.set_defaults(fn=cmd_probe)
    r = sub.add_parser("record")
    r.add_argument("--config", default=str(ROOT / "configs" / "capture.yaml"))
    r.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl+C)")
    r.add_argument("--export-last", type=float, default=None, help="export the last N seconds when stopping")
    r.add_argument("--status-every", type=float, default=5.0)
    r.set_defaults(fn=cmd_record)
    b = sub.add_parser("benchmark")
    b.add_argument("--seconds", type=float, default=6.0)
    b.set_defaults(fn=cmd_benchmark)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
