# Evaluation

Every number here was measured in this repository on 2026-10-12. The raw outputs are in `outputs/reports/`. **None of these numbers establishes real-world DRS accuracy.**

## 1. Automated tests

- `pytest`: 71 tests (59 ML, 12 API), all passing. They cover:
  - video metadata, frame extraction, and empty or invalid files;
  - the detection schema, NMS, tiling, the motion detector and missing-weights errors;
  - Kalman initialisation, missing frames, outlier rejection, static clutter and camera cuts;
  - 2D fits, change points and intervals;
  - 3D ballistic fits and monocular observability;
  - calibration validation and mislabelled points;
  - triangulation with known geometry, sync offsets, inconsistent views and unsynchronised cameras;
  - LBW hit, miss, over-the-top, marginal, unvalidated-calibration and rules cases, plus uncertainty propagation;
  - dataset splits, detection metrics, and API validation, jobs, cancellation, Range requests, exports and PDF.
- Frontend: `tsc --noEmit`, ESLint and `next build` all clean.
- Ruff: clean.

Synthetic tests verify behaviour. They do not measure cricket accuracy.

## 2. Trajectory and LBW projection on cricket-synth (synthetic)

**Script:** `scripts/evaluate_trajectory_synth.py`.

**Data:** `Ranvin36/cricket-synth` (MIT). The deliveries are simulated with drag, Magnus, seam and a friction bounce, which is richer physics than the fitted model. The rig is `front_on_stump`, behind the bowler's stumps, at 60 fps and 1280×720.

**Protocol:**
- A virtual pad is placed 1.5 m in front of the stumps.
- Only pixels *before* that point are used, from ground-truth centres plus 1 px Gaussian noise. The apparent radius has 0.5 px noise.
- The camera is **exactly known** (validated).
- The LBW engine uses its defaults: 95% decision confidence and covariance inflation ×6.

| | Calibration split (clips 0–119) | **Held-out split (clips 120–239)** |
|---|---|---|
| Clips with a stump-plane projection | 99 / 120 | 108 / 120 |
| Stump-plane error, median | 6.8 cm | **7.9 cm** |
| Stump-plane error, p90 | 24.6 cm | 38.4 cm |
| Stump-plane error, max | 12.4 m | 5.7 m |
| Empirical coverage of the 90% interval | 0.91 | **0.89** |
| Inconclusive or insufficient-data rate | 0.53 | 0.54 |
| Definite verdicts | 56 (all correct) | **55 (all correct)** |
| False "hit" / false "miss" | 0 / 0 | 0 / 0 |
| True hits called `PROJECTED_TO_HIT` | 0 of 15 | **0 of 20** |

**How the inflation factor was chosen.** The covariance inflation was tuned on the calibration split. The settings tried were ×1, ×4 and ×9, which gave 90% coverage of 0.65, 0.86 and 0.93 respectively, with inconclusive rates of 0.30, 0.46 and 0.63, and no definite errors at any setting. ×6 was then checked once on the held-out split; reproduce any setting with `--inflation`.

**Interpretation:**
- The geometry and uncertainty machinery behaves as intended on synthetic data: intervals are calibrated and there were no confident errors.
- The engine is **conservative**. It never confirmed a hit in either split, and about half the deliveries come out inconclusive.
- The large-error tail comes from deliveries whose post-bounce depth is poorly observable from a single camera.
- This benchmark uses an exact camera and ideal detections. Real footage adds calibration error, missed and false detections, motion blur and lower frame rates.

**Design changes made because of this benchmark** (kept in the code, documented in `ml/trajectory/ballistic.py`):
- A gravity-only model biased depth by 2–5 m, so drag (Cd ≈ 0.4) and a fitted lift term were added.
- The apparent ball size is used as a depth cue.
- A weak prior on speed retention through the bounce was added.
- A regularised covariance replaced `pinv`, which had reported unobservable parameters as certain.

## 3. Detection and tracking on real broadcast footage (qualitative, unlabelled)

**Data:** 5 clips from `taha1418/Cricket_Delivery_Videos` (1920×1080, 25 fps, broadcast). There are **no ground-truth labels**, so no precision or recall can be computed.

**Detector:** generic COCO YOLO11s "sports ball" combined with motion candidates (`coco_hybrid`), with shot-cut handling.

| Clip | Primary track observed / spanned | Tracker consistency check |
|---|---|---|
| video001 | 34 / 40 | unreliable (mean innovation 72 px) |
| video002 | 99 / 147 | unreliable (279 px) |
| video003 | 4 / 4 | unreliable (too few observations) |
| video004 | 82 / 152 | unreliable (616 px) |
| video005 | 32 / 38 | unreliable (68 px) |

What visual inspection showed:
- The COCO model detects the ball in close-up shots (for example 0.90 confidence during a catch).
- It also fires on static TV logos (0.49–0.69).
- It misses the few-pixel ball in wide shots.
- Motion candidates are dominated by players.

**Conclusion:** real-footage ball tracking is **not working yet**. That is the expected result without a cricket-trained detector, and the reliability flag correctly marks these tracks as untrustworthy.

## 4. Detector training pipeline (smoke test)

`train_detector.py --smoke` was run on `data/processed/synth_yolo`: 110 synthetic frames from 3 clips, split by clip into 46 train, 18 val and 46 test. It used the RTX 3080 Ti in Docker, ran 2 epochs at 640 px, and took 23 s.

Test-split results (`detector_eval_synth_smoke_test.json`): precision 0.00, recall 0.00, AP (centre match) 0.04. The median ground-truth ball size is 7.2 px.

This verifies that training, checkpointing, model cards and evaluation run end to end. **The model is not usable** and was not promoted.

## 4b. Cricket-ball detector (YOLO11s fine-tuned on the Kaggle CC0 set)

- **Leakage found and fixed.** The dataset's own splits share 112 identical files between train and val/test, so scores on its official test split would be inflated. All splits were rebuilt with `--resplit`, which groups images by filename and dHash before splitting.
- **Training:** 60 epochs at 1280 px took 28.5 min. Validation precision 0.90, recall 0.91, mAP50 0.94.

| Test set | P | R | F1 | AP |
|---|---|---|---|---|
| Kaggle held-out test (179 images, 188 balls, median 69 px) | 0.91 | 0.98 | 0.94 | 0.99 |
| Independent broadcast set (298 frames, 213 balls, median 13 px), full frame | 0.64 | 0.37 | 0.47 | 0.38 |
| Same, tiled 640 | 0.46 | 0.48 | 0.47 | 0.37 |
| Same, tiled 960 | 0.43 | 0.50 | 0.46 | 0.36 |

On the same 5 unlabelled broadcast clips as in section 3:
- Candidate tracks fell from 80–130 per clip to 3–8.
- **2 of 5** primary tracks pass the consistency checks. video003 was visually confirmed to be the real ball (17 of 18 frames).
- 3 of 5 are still unreliable.

**Conclusion:** the in-domain detector is good, but detection of the small ball in broadcast footage remains weak (recall 0.37). The fix is broadcast-domain training data.

## 5. GPU inference throughput (YOLO11s, 1080p input, RTX 3080 Ti)

| imgsz | batch | precision | Docker (Linux) fps | Native Windows fps | Peak VRAM |
|---|---|---|---|---|---|
| 640 | 16 | FP16 | 455 | 431 | 0.29 GB |
| 1280 | 1 | FP32 | 110 | 109 | 0.21 GB |
| 1280 | 8 | FP16 | 168 | 69* | 0.51 GB |
| 1280 | 16 | FP16 | 176 | 163 | 0.87 GB |
| 1280 | 16 | FP32 | 132 | 125 | 1.72 GB |

\* The native run shared the CPU with two benchmark processes, so some mid-batch numbers are noisy. The full tables are in `inference_benchmark_*.json`.

- Decoding runs at 245 fps (Docker) to 770 fps (native) for 1080p H.264.
- **End-to-end throughput** for hybrid detection, tracking and 1080p annotated-video rendering is about 10 fps in Docker. With the motion heuristic only it is about 18 fps (native CPU).
- The pipeline is offline and not real-time. Rendering and motion candidates, not the GPU, are the bottleneck.

## 6. Not measured (no ground truth available)

- Detection metrics on real cricket footage.
- Tracking continuity and fragmentation against labelled real sequences.
- 3D error on real deliveries.
- LBW accuracy against reference decisions.
- Real multi-camera synchronisation sensitivity.
- Contact detection.
