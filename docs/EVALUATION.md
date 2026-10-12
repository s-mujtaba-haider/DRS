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
- The LBW engine uses its current defaults:
  - 95% decision confidence;
  - flight model with drag, fitted lift (prior σ 3 m/s²) and fitted lateral swing/drift (prior σ 2 m/s²);
  - covariance inflation ×2.

| | v1 model (no swing, ×6), held-out | **Current model, calibration split (0–119)** | **Current model, held-out (120–239)** |
|---|---|---|---|
| Clips with a stump-plane projection | 108 / 120 | 107 / 120 | **111 / 120** |
| Stump-plane error, median | 7.9 cm | 6.4 cm | **5.7 cm** |
| Stump-plane error, p90 | 38.4 cm | 16.4 cm | **26.0 cm** |
| Stump-plane error, max | 5.7 m | 0.95 m | 4.6 m |
| Empirical coverage of the 90% interval | 0.89 | 0.93 | **0.87** |
| Inconclusive or insufficient-data rate | 0.54 | 0.31 | **0.39** |
| Definite verdicts (all correct) | 55 | 82 | **73** |
| False "hit" / false "miss" | 0 / 0 | 0 / 0 | **0 / 0** |
| True hits called `PROJECTED_TO_HIT` | 0 of 20 | 1 of 15 | 1 of 20 |

**How the settings were chosen:** on the calibration split only, then checked once on the held-out split.
- The swing term was added after the v1 model showed a large error tail.
- Inflation with the swing model:

| Inflation | 90% coverage | Inconclusive rate |
|---|---|---|
| ×1 | 0.79 | 0.23 |
| ×2 | 0.93 | 0.31 |
| ×3 | 0.94 | 0.38 |

- No definite errors occurred at any setting.
- Reproduce any setting with `--inflation` and `--swing-sigma` (`0` disables the swing term).

**Interpretation:**
- Modelling lateral swing/drift cut the p90 error by about a third and turned a quarter more deliveries into correct definite verdicts, still with zero confident errors.
- Coverage on the held-out split (0.87) is slightly below nominal. The intervals are close to, but not fully, calibrated.
- The engine is **still conservative on hits**: only 1 of 20 true hits is confirmed. Hits sit close to the stump edges more often, and a single camera cannot resolve them at 95% confidence.
- This benchmark uses an exact camera and ideal detections. Real footage adds calibration error, missed and false detections, motion blur and lower frame rates.

**Design changes made because of this benchmark** (kept in the code, documented in `ml/trajectory/ballistic.py`):
- A gravity-only model biased depth by 2–5 m, so drag (Cd ≈ 0.4) and a fitted lift term were added.
- The apparent ball size is used as a depth cue.
- A weak prior on speed retention through the bounce was added.
- A regularised covariance replaced `pinv`, which had reported unobservable parameters as certain.
- A lateral swing/drift acceleration with a prior was added.

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

## 4d. Self-training on real broadcast footage (v2)

**Method** (`scripts/pseudo_label_broadcast.py`):
1. The v1 detector and the tracker are run over the 214 allowed broadcast clips. Zimbabwe series are excluded because they could overlap the test match.
2. Detections that belong to **consistently moving tracks** become labels. A track qualifies with ≥ 8 observations, mean Kalman innovation ≤ 9 px and mean confidence ≥ 0.25.
3. The result is 1,968 labels on 1,727 native-resolution crops, split by series: New Zealand v Pakistan for validation, the rest for training. A random sample of 48 crops was inspected and all showed a real ball.
4. v1 was fine-tuned on Kaggle plus these labels at 960 px for 50 epochs.

**Independent human-labelled broadcast test** (298 frames, 213 balls, IND v ZIM 2022; never used for training):

| Model | Inference | P | R | F1 | AP |
|---|---|---|---|---|---|
| v1 (Kaggle only) | 1280 | 0.64 | 0.37 | 0.47 | 0.38 |
| **v2 (Kaggle + track-verified broadcast)** | **1280** | **0.62** | **0.53** | **0.57** | **0.49** |
| v2 | 1920 | 0.47 | 0.58 | 0.52 | 0.53 |
| v2 | 960 | 0.62 | 0.42 | 0.50 | 0.42 |

- In-domain Kaggle test for v2 at 1280: P 0.88, R 0.98, AP 0.93 (v1: 0.91 / 0.98 / 0.99).
- Self-training gives a real-domain gain of **+42% recall and +29% AP** at about the same precision.
- **Caveat:** pseudo-labels favour balls the previous model could already detect (self-training bias), and the test set is a single match.

## 4e. Second self-training round (v3, the current detector)

**Method:** the same as §4d, but the pseudo-labels came from v2.
- v2 plus the tracker gave 3,373 labels on 3,273 crops from 213 clips (196 verified tracks).
- v1 was fine-tuned on Kaggle plus these labels at 960 px for 50 epochs (43 min).

**Label quality dropped.** A random sample of 48 crops held about 33 clear balls. About 8 were clearly something else: a fielder's cap, a logo letter, stumps or crease marks, and shoes near the stumps. About 7 were unclear. In round 1, all 48 sampled crops were balls.

**Independent human-labelled broadcast test** (same 213 balls as §4d):

| Model | Inference | P | R | F1 | AP | FP |
|---|---|---|---|---|---|---|
| v2 | 960 | 0.62 | 0.42 | 0.50 | 0.42 | 56 |
| v3 | 960 | 0.65 | 0.44 | 0.53 | 0.45 | 51 |
| v2 | 1280 | 0.62 | 0.53 | 0.57 | 0.49 | 70 |
| **v3** | **1280** | **0.63** | **0.51** | **0.57** | **0.53** | **63** |
| v2 | 1920 | 0.47 | 0.58 | 0.52 | 0.53 | 141 |
| v3 | 1920 | 0.48 | 0.57 | 0.52 | 0.54 | 132 |

- In-domain Kaggle test at 1280: v3 P 0.87, R 0.97, AP 0.96 (v2: 0.88 / 0.98 / 0.92).
- At 1280, F1 is unchanged and AP is +0.04. The differences are a handful of detections (−3 TP, −7 FP) on a single match, which is within noise.

**Tracking on the 5 reference clips** (score 0.15, 1280 px):
- Reliable primary tracks: **v3 3 of 5** (video002, video004, video005); v2 2 of 5 (video001, video004).
- All five reliable tracks were checked visually, and every one is the real ball. In every case it is the ball **after** the shot: high in the air, or running across the outfield to the rope. None is the delivery.
- Caveat: these clips were in the pseudo-labelling pool, so this check is not held out.

**Decision:** v3 was promoted. It is level or better on every measure, but the gain is small.

**Conclusion:**
- Self-training has plateaued, and the label noise is growing.
- A third round would need stricter track filters, for example rejecting tracks that start near the stumps or players, or requiring larger image-plane motion.
- The better next step is human-labelled broadcast frames (Roadmap 1).

## 4c. Experiments that did not help (kept for the record)

**Copy-paste compositing (`scripts/build_broadcast_training_set.py`, `ml/datasets/composite.py`).**
- Method:
  - Kaggle training-split balls were shrunk to 5–22 px, motion-blurred and colour-jittered.
  - They were pasted onto 1,944 real frames from 214 broadcast clips. The excluded series were Zimbabwe matches; validation used New Zealand v Pakistan.
  - The result was merged with the Kaggle set and YOLO11s was trained at native 960 px crops for 60 epochs.
- Validation mAP50 on composites reached 0.89, but on the real independent broadcast set it got **worse**:

| Inference | P | R | F1 | AP |
|---|---|---|---|---|
| 1280 | 0.44 | 0.46 | 0.45 | 0.33 |
| 1920 (native) | 0.36 | 0.40 | 0.38 | 0.26 |

- Baseline at 1280 for comparison: 0.64 / 0.37 / 0.47 / 0.38.
- Interpretation: pasted balls are easier to separate from the background than real ones, so the model learns "bright compact blob" and fires on more real distractors.
- **Not promoted.** The code is kept so the result can be reproduced.

**Tracker hysteresis with a lower detection threshold.**
- Setting tried: detector score ≥ 0.05; spawn a track only at ≥ 0.25; extend it with ≥ 0.05.
- Reliable ball tracks on the 5 reference clips **dropped from 2 to 1**, and the ball track's mean innovation rose from 6.6 px to 10.8 px. Weak false detections were absorbed into tracks, and real low-confidence balls (about 0.25) could no longer start a track.
- Detection thresholds of 0.10 and 0.15 with spawn 0.25 also gave 1 of 5.
- The defaults stay at score 0.15 with no hysteresis. The option remains in `TrackerConfig`.

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
