# Cricket DRS — research prototype

An independent, locally-run research system for cricket ball tracking and **experimental** decision-review analysis (LBW first). It ingests video, detects and tracks the ball, fits physically-informed trajectories with explicit uncertainty, and presents everything in a web review console.

> **This is not an official decision system.** It is not certified, has not been validated against professional ball-tracking, and its real-world accuracy is **unknown**. Every result carries a disclaimer and its uncertainty, and the engine returns `INCONCLUSIVE` rather than guess.

## Current status at a glance

| Area | Status |
|---|---|
| Video ingestion (metadata, PTS timestamps, VFR detection, frame extraction) | Implemented and tested |
| Ball detection | Pipeline implemented and tested. **No cricket-trained detector yet** (training data needs a Kaggle/Roboflow key, see below). Available now: generic COCO "sports ball" YOLO11 and a motion heuristic. Neither tracks the ball reliably in broadcast footage (measured, see `docs/EVALUATION.md`). |
| Ball tracking (Kalman CA, gated Hungarian, RTS smoothing, fragment merge, camera-cut handling, reliability flag) | Implemented and tested on synthetic data. Real-footage quality limited by detection. |
| 2D trajectory (robust polynomial + change point, BIC/LOO, prediction intervals) | Implemented and tested |
| Camera calibration (chessboard intrinsics, PnP from pitch landmarks, approximate mode) | Implemented and tested |
| Multi-view triangulation (sync association, consensus, covariance) | Implemented and tested on synthetic geometry. No real multi-view data processed. |
| 3D flight model (gravity, drag, lift, bounce; monocular and multi-view fits) | Implemented. Benchmarked on synthetic deliveries. |
| Experimental LBW engine (Monte Carlo, Law 36 rules layer, unknowns stay unknown) | Implemented. Benchmarked on synthetic data only. |
| Contact (edge) analysis | Experimental: trajectory-deflection candidates only. Returns `INCONCLUSIVE` below 100 fps. |
| Run-out, stumping, boundary | **Planned.** Not implemented (see `docs/ROADMAP.md`). |
| Web console (dashboard, upload/jobs, frame-accurate review, corrections, calibration, LBW, 3D, reports) | Implemented. Type-checked, linted and production-built. |
| Backend API (FastAPI, background jobs, SQLite, PDF/JSON/CSV exports) | Implemented and tested |

## Architecture

```
frontend/  Next.js 16 + React 19 + TypeScript + Tailwind, three.js (r3f), recharts   :3100
   |  REST (JSON, multipart upload, Range video streaming)
backend/   FastAPI + SQLAlchemy (SQLite) + background job manager                    :8100
   |  python calls
ml/        video | detection | tracking | trajectory | calibration | reconstruction | lbw | contact_analysis | evaluation | datasets
scripts/   CLI: download/validate/prepare datasets, train/evaluate detector, run tracking/analysis, calibrate, benchmark
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Hardware

Developed and measured on an RTX 3080 Ti (12 GB), Ryzen 7 7700, 16 GB RAM, Windows 11.

- **GPU inference:** YOLO11s at 1280 px runs at 160–175 frames/s with batch 16 in FP16, using under 1 GB of VRAM.
- **CPU-only:** works with the motion heuristic, at about 18 frames/s for end-to-end 1080p processing including rendering.

## Quick start (Windows, PowerShell)

```powershell
git clone https://github.com/s-mujtaba-haider/DRS.git
cd DRS
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements\backend.txt -r requirements\ml-gpu.txt -r requirements\dev.txt
# (exact versions used during development: requirements\lock-windows.txt)

# datasets that download anonymously (Hugging Face)
.\.venv\Scripts\python scripts\download_datasets.py
# generic COCO weights for the fallback detector (AGPL-3.0, NOT cricket-trained)
.\.venv\Scripts\python -c "from ultralytics import YOLO; YOLO('models/checkpoints/yolo11s.pt')"

# backend  -> http://127.0.0.1:8100/docs
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8100

# frontend (second terminal) -> http://localhost:3100
cd frontend; npm install; npm run build; npm run start
```

Full instructions, Docker GPU option and troubleshooting are in [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

> **Windows Smart App Control.** On first use, Smart App Control may block PyTorch's unsigned DLLs (`WinError 4551`). This happened during development, and the block cleared later. If it persists, use the Docker GPU backend (`docker compose up --build`). It runs the same code in Linux with CUDA. The app also keeps working without PyTorch, falling back to the motion-heuristic detector.

## Using the console

1. **New analysis.** Upload a video, inspect its metadata and warnings, choose a detector, and start. Progress can be cancelled.
2. **Session › Tracking.**
   - Frame-accurate playback (`←`/`→` step, space play/pause, 0.1–1× speed) with overlays.
   - Overlay colours: **green** observed, **amber dashed** interpolated, **cyan** smoothed, **blue** manual, **grey** other candidates.
   - The timeline shows observed and missing frames; there is also a confidence chart and track-quality metrics with a reliability flag.
   - Corrections: *Correct ball* places a manual position and *Reject detection* rejects one. Corrections are stored separately and never overwrite model output.
   - *Set as impact frame* marks the frame where the ball meets the pad.
3. **Session › LBW.**
   - Calibrate by picking named pitch landmarks and clicking them on an exact decoded frame (≥ 6 points without known intrinsics). The result shows reprojection error, assumptions and warnings.
   - Enter the facts the system can't determine (handedness, no ball, bat first, shot offered) and run the analysis.
   - The output includes a verdict, P(hit), the Law 36 checks, the stump-plane view with its 2σ region, and an overlay of observed, fitted and predicted paths.
4. **Session › 3D.** A Three.js scene of the pitch, stumps, estimated camera and fitted or predicted path. A banner marks single-camera results as *approximate*.
5. **Session › Report.** Downloads: PDF report, JSON export, CSV track points, annotated MP4 and raw detections.

## CLI

```powershell
python scripts\inspect_environment.py
python scripts\download_datasets.py [--only ID]       # registry-driven; records failures
python scripts\validate_datasets.py [--viz 12]
python scripts\prepare_datasets.py synth | yolo-dir PATH --name NAME
python scripts\annotate_video.py export|from-ui|build ...
python scripts\train_detector.py --data DATA.yaml [--smoke] [--promote]
python scripts\evaluate_detector.py --data DATA.yaml --split test --weights W.pt
python scripts\run_tracking.py VIDEO --out DIR [--detector auto|cricket|coco_hybrid|coco|motion]
python scripts\calibrate_cameras.py intrinsics|pose|landmarks ...
python scripts\run_drs_analysis.py VIDEO --out DIR --camera camera.json --impact-frame N --handedness right
python scripts\evaluate_trajectory_synth.py --limit 120 [--offset 120]
python scripts\benchmark_inference.py VIDEO
```

## Getting a real cricket ball detector (the main open item)

The most useful verified dataset, **Cricket Ball Dataset for YOLO**, has 1,778 images under CC0 and needs a Kaggle API token:

1. Go to kaggle.com, then Settings, then API, and click *Create New Token*. Save the downloaded `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json`.
2. `python scripts\download_datasets.py --only kaggle-kushagra-ball-yolo`
3. `python scripts\prepare_datasets.py yolo-dir data\raw\kaggle-kushagra-ball-yolo --name ball_kaggle --origin human`
4. `python scripts\train_detector.py --data data\processed\ball_kaggle\data.yaml --smoke` (pipeline check), then a baseline run without `--smoke`
5. `python scripts\evaluate_detector.py ...` on the test split, **and** on frames you label from your own target footage (`annotate_video.py`).
6. Use `--promote` only after that evaluation. It copies the model to `models/checkpoints/cricket_ball_best.pt`, which the app then uses automatically.

## Tests

```powershell
.\.venv\Scripts\python -m pytest            # ML + API: 70+ tests (synthetic fixtures)
.\.venv\Scripts\ruff check ml backend scripts tests
cd frontend; npx tsc --noEmit; npm run lint; npm run build
```

Synthetic tests verify that the implementation behaves as designed. They don't establish accuracy on real cricket footage.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Installation](docs/INSTALLATION.md)
- [Datasets](docs/DATASETS.md)
- [Model card](docs/MODEL_CARD.md)
- [Evaluation](docs/EVALUATION.md)
- [Limitations](docs/LIMITATIONS.md)
- [Roadmap](docs/ROADMAP.md)

## Licence

Project code: MIT. Third-party components keep their own licences:
- Ultralytics YOLO is **AGPL-3.0**, which matters if you distribute or serve a modified system.
- Datasets carry their own terms; see `docs/DATASETS.md`.
