# Cricket DRS — research prototype

An independent, locally-run research system for cricket ball tracking and **experimental** decision-review analysis (LBW first). It ingests video, detects and tracks the ball, fits physically-informed trajectories with explicit uncertainty, and presents everything in a web review console.

> **This is not an official decision system.** It is not certified, has not been validated against professional ball-tracking, and its real-world accuracy is **unknown**. Every result carries a disclaimer and its uncertainty, and the engine returns `INCONCLUSIVE` rather than guess.

## Current status at a glance

| Area | Status |
|---|---|
| Video ingestion (metadata, PTS timestamps, VFR detection, frame extraction) | Implemented and tested |
| Ball detection | **Cricket-trained YOLO11s v3**: trained on the CC0 Kaggle set, then self-trained for two rounds on real broadcast clips using track-verified pseudo-labels. **Independent broadcast frames: P 0.63 / R 0.51 / AP 0.53** (v2: 0.62 / 0.53 / 0.49; v1: 0.64 / 0.37 / 0.38). About half of tiny broadcast balls are still missed. See `docs/EVALUATION.md` §4. |
| Ball tracking (Kalman CA, gated Hungarian, RTS smoothing, fragment merge, camera-cut handling, reliability flag) | Implemented and tested on synthetic data. Real-footage quality limited by detection. |
| 2D trajectory (robust polynomial + change point, BIC/LOO, prediction intervals) | Implemented and tested |
| Camera calibration (chessboard intrinsics, PnP from pitch landmarks, approximate mode) | Implemented and tested |
| Automatic calibration seed from detected stumps (CLI `calibrate_cameras.py auto`, *Auto-detect* in the web calibration panel) | **Experimental.** Stumps detector trained on one match (held-out P 0.58 / R 0.58). On 40 clips from other matches, 3 calibrated, all visually correct; 37 refused. Never marked validated. See `docs/EVALUATION.md` §4f. |
| Multi-view triangulation (sync association, consensus, covariance) | Implemented and tested on synthetic geometry. No real multi-view data processed. |
| 3D flight model (gravity, drag, lift, bounce; monocular and multi-view fits) | Implemented. Benchmarked on synthetic deliveries. |
| Experimental LBW engine (Monte Carlo, Law 36 rules layer, unknowns stay unknown) | Implemented. Benchmarked on synthetic data only. |
| Contact (edge) analysis | Experimental: trajectory-deflection candidates only. Returns `INCONCLUSIVE` below 100 fps. |
| Run-out and stumping (crease geometry, wicket-break candidates, frame-interval timing logic, Laws 29/30/38/39) | Implemented. Tested on synthetic geometry and footage. Evidence comes from operator-marked frames; the engine says `INCONCLUSIVE` when grounding and the break fall in the same inter-frame gap. |
| Boundary and catch-near-boundary (rope geometry, event sequence, Law 19 incl. 2026 19.5.2, Law 33) | Implemented. Tested on synthetic sequences. Image-space rope geometry is valid only for ground contacts. |
| DRS protocol engine (versioned ICC Test/ODI/T20I 2025 and club configs; umpire's call, review counts) | Implemented and tested. Some protocol values are from secondary sources and are listed as `unverified` in each config. |
| Match-day console (matches, reviews, evidence, advisory recommendation, human decision, audit log, evidence package) | Implemented and tested, and checked in the browser end to end. See `docs/MATCH_DAY.md`. |
| Live multi-camera capture (rolling disk buffer, NVENC, preview, instant clip export with per-frame capture timing) | Implemented. Measured: 3 × 1080p100 on this PC with NVENC in process mode (`docs/EVALUATION.md` §7). **Software sync only.** |
| Multi-camera time sync (audio GCC-PHAT, visual-event interval matching) | Implemented. Tested on synthetic signals. Not validated on real multi-camera footage. |
| Multi-camera LBW (per-camera tracking → time alignment → triangulation → 3D flight fit, sync uncertainty propagated) | Implemented. Synthetic: with hardware sync 54/60 definite verdicts, all correct, and 18/22 true hits confirmed (one camera: 1/20). Software sync always `INCONCLUSIVE` by design (`docs/EVALUATION.md` §10). |
| Match-day operation extras: synchronised multi-camera replay, second-screen umpire monitor, rehearsal (replay a video as a live camera), PDF review report with annotated evidence frames, operator/umpire PINs | Implemented and tested. |
| Batter foot proposals for run-outs (YOLO11s-pose, AGPL-3.0, not cricket-trained) | Experimental. On 40 broadcast clips the batter was selected in 17; localisation accuracy unmeasured (`docs/EVALUATION.md` §9). |
| Wicket-break detection on a panning/zooming camera (homography compensation) | Implemented. Tested on a synthetic panning clip. |
| Ball-labelling page (detector suggestions, keyboard workflow, YOLO export split by video) | Implemented. This is the route to better broadcast detection. |
| Web console (dashboard, upload/jobs, frame-accurate review, corrections, calibration, LBW, 3D, reports) | Implemented. Type-checked, linted and production-built. |
| Backend API (FastAPI, background jobs, SQLite, PDF/JSON/CSV exports) | Implemented and tested |

## Architecture

```
frontend/  Next.js 16 + React 19 + TypeScript + Tailwind, three.js (r3f), recharts   :3100
   |  REST (JSON, multipart upload, Range video streaming)
backend/   FastAPI + SQLAlchemy (SQLite) + background job manager                    :8100
   |  python calls
ml/        video | capture | synchronization | detection | tracking | trajectory | calibration | reconstruction
           lbw | runout | boundary | protocol | contact_analysis | evaluation | datasets
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

**At a match:** follow [`docs/MATCH_DAY.md`](docs/MATCH_DAY.md). In short: *Live capture* starts the cameras, *Match day* creates the match, and each review clips the last seconds of every camera. The operator marks the evidence and computes the advisory recommendation; the third umpire records the decision.

**Offline analysis of a single clip:**


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
python scripts\calibrate_cameras.py intrinsics|pose|landmarks|auto ...
python scripts\run_drs_analysis.py VIDEO --out DIR --camera camera.json --impact-frame N --handedness right
python scripts\evaluate_trajectory_synth.py --limit 120 [--offset 120]
python scripts\benchmark_inference.py VIDEO
python scripts\capture_cameras.py probe|record|benchmark ...    # live capture without the web console
python scripts\estimate_sync.py audio|events ... [--apply-to camera.json]
python scripts\evaluate_sync_sensitivity.py
python scripts\analyze_runout.py detect|analyze ...
```

## The cricket ball detector

Weights are not committed. Reproduce them (about 30 min on an RTX 3080 Ti) with the commands in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

Kaggle downloads need an API token. Set `KAGGLE_API_TOKEN`, or save the token to `%USERPROFILE%\.kaggle\access_token`. **Never commit it.**

The biggest remaining gain is **broadcast-domain training data**:
1. Label frames from your own footage with `annotate_video.py`, plus UI corrections.
2. Build a dataset split by video, retrain, and re-evaluate on `broadcast_seg_test`, which must stay test-only.

## Tests

```powershell
.\.venv\Scripts\python -m pytest            # ML + API (synthetic fixtures)
.\.venv\Scripts\ruff check ml backend scripts tests
cd frontend; npx tsc --noEmit; npm run lint; npm run build
```

Synthetic tests verify that the implementation behaves as designed. They don't establish accuracy on real cricket footage.

## Documentation

- [Match-day operation](docs/MATCH_DAY.md)
- [International DRS requirements](docs/INTERNATIONAL_DRS_REQUIREMENTS.md)
- [Camera hardware](docs/CAMERA_HARDWARE.md)
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
