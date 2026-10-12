# Architecture

## Layers

| Layer | Path | Responsibility |
|---|---|---|
| Shared schemas | `ml/common/schemas.py` | Pydantic contracts: `VideoMetadata`, `Detection`, `TrackPoint` (with `kind`), `BallTrack`, `TrackingResult` |
| Ingestion | `ml/video/ingest.py`, `ml/video/shots.py` | PyAV decoding, PTS timestamps, VFR detection, frame extraction, H.264 writer, camera-cut detection |
| Detection | `ml/detection/` | `BallDetector` protocol, plus three implementations: Ultralytics YOLO (batching, tiling, FP16, OOM back-off, CPU fallback), motion candidates (3-frame differencing with phase-correlation compensation), and a hybrid of the two |
| Tracking | `ml/tracking/` | Constant-acceleration Kalman filter with RTS smoother, gated Hungarian association, multi-hypothesis, fragment merging, outlier rejection, reliability checks |
| 2D trajectory | `ml/trajectory/image_trajectory.py` | Huber-robust polynomials with an optional change point. Selection by BIC, validation by LOO, prediction intervals. |
| 3D flight | `ml/trajectory/ballistic.py` | Gravity, drag, lift, and bounce at ground contact. Fits to 3D points or monocular pixels. Covariance and Monte Carlo sampling. |
| Calibration | `ml/calibration/` | Pitch frame and Law 7/8 dimensions, named landmarks, chessboard intrinsics, PnP pose, approximate mode, `CameraModel` import/export |
| Reconstruction | `ml/reconstruction/triangulation.py` | Time-synchronised association, DLT plus Gauss–Newton, view-consensus rejection, ray-angle conditioning, covariance |
| LBW | `ml/lbw/` | `geometry.py` (pure geometry, Monte Carlo), `rules.py` (Law 36 checks), `engine.py` (verdict, downgrades, uncertainty) |
| Contact | `ml/contact_analysis/deflection.py` | Trajectory-deflection candidates with alternative explanations |
| Evaluation | `ml/evaluation/` | Synthetic generator, cricket-synth benchmark, detection metrics |
| Datasets | `ml/datasets/yolo_tools.py` | Validation, dHash duplicate detection, sequence-level splits, provenance, cricket-synth conversion |
| Pipeline | `ml/pipeline.py` | Streaming end-to-end run with progress and cancellation hooks |
| Backend | `backend/app/` | FastAPI routers (`system`, `videos`, `jobs`, `analysis`), SQLAlchemy models, `JobManager` thread pool, services (LBW glue, PDF) |
| Frontend | `frontend/` | Next.js app router pages plus components (`TrackingWorkspace`, `LbwWorkspace`, `Trajectory3D`) |

The frontend only talks to the REST API (`frontend/lib/api.ts`). It never touches ML internals.

## Coordinate frame

The pitch frame is measured in metres:
- **Origin:** ground level at the base of the striker's middle stump.
- **+y:** towards the bowler.
- **+z:** up.
- **+x:** completes a right-handed frame, so it points to the bowler's left. That is the off side for a right-handed batter.

This frame matches the cricket-synth dataset after shifting y by half the pitch length (10.06 m).

## Data flow for one video

1. `POST /api/videos` streams the upload to `storage/videos/<uuid>.<ext>`. The server never uses client paths. It enforces a size limit, computes a SHA-256, and probes the file with PyAV; undecodable files are rejected.
2. `POST /api/analysis/jobs` stores the config and queues the job in the `JobManager`, which uses one worker so GPU memory stays bounded. The request returns immediately.
3. The worker runs `run_pipeline` in streaming fashion:
   1. Decode the frames and detect cuts, then run detection in batches.
   2. Run tracking and fit the 2D trajectory.
   3. Write JSON and CSV outputs, then render `annotated.mp4` in a second pass.
   4. Persist progress and status in SQLite. Cancelling sets an event that is checked per frame.
4. The UI reads `/results` and `/detections`, and streams the original video via HTTP Range. It draws overlays client-side, synced per presented frame (`requestVideoFrameCallback`).
5. Calibration (`POST /api/calibration`) stores the `CameraModel` together with the raw correspondences, so it can be reproduced.
6. `POST /api/lbw/analyze` runs these steps:
   1. Read the primary-track observations and apply manual corrections in memory.
   2. Fit the monocular flight model (`fit_to_pixels`), then run `analyze_lbw`.
   3. Store the request and result, plus overlay geometry projected back into the image.
7. Reports and exports are built from the stored values only.

## Observation vs estimate

`TrackPoint.kind` is one of `observed`, `filtered`, `interpolated`, `predicted` or `manual`. The UI, CSV, annotated video and 3D view all use the same colour code. Manual annotations live in a separate table and never modify the job's stored outputs.

## Extending

- **A new detector:** implement `detect_batch(images, infos) -> list[list[Detection]]` and register it in `ml/pipeline.build_detector`.
- **A new decision module** (run-out, stumping, boundary): add `ml/<module>/` with a geometry/evidence layer and a rules layer, as LBW does. Then add a backend router and a UI tab.
- **Multi-camera ingestion:** the calibration and triangulation layers already support it (`/api/reconstruction/triangulate`). What's missing is per-camera job orchestration and synchronisation estimation (see the roadmap).
