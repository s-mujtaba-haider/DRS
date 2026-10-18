# Evaluation

Every number here was measured in this repository on 2026-10-12. The raw outputs are in `outputs/reports/`. **None of these numbers establishes real-world DRS accuracy.**

## 1. Automated tests

- `pytest`: 215 tests, all passing (2026-10-18). Besides the areas below, they cover the match/review API, PIN roles, protocol engine, run-out/stumping/boundary engines and endpoints, wicket-break motion compensation, foot proposals, multi-view LBW, live capture and rehearsal replay through the API, synchronisation, labelling export and PDF review reports. The earlier set covers:
  - video metadata, frame extraction, and empty or invalid files;
  - the detection schema, NMS, tiling, the motion detector and missing-weights errors;
  - Kalman initialisation, missing frames, outlier rejection, static clutter and camera cuts;
  - 2D fits, change points and intervals;
  - 3D ballistic fits and monocular observability;
  - calibration validation and mislabelled points, and calibration from stumps boxes (wicket pairing, focal grid, implausible cameras);
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

## 4f. Stumps detector and automatic calibration seed (experimental)

**Data and training.**
- Stumps boxes come from the Acme AI masks (`prepare_datasets.py stumps-seg`), which cover one match (IND v ZIM 2022).
- The set is split by match time: frames ≤ 175 train (130 boxes), 176–260 val (81), ≥ 261 test (73, later in the second innings).
- The ball detector never trains on these frames; only this separate stumps model does.
- YOLO11s (COCO) was trained at 1280 px for 100 epochs.

**Detector, held-out test frames** (`stumps_detector_eval_y11s_1280.json`, conf 0.25, IoU ≥ 0.5):

| Split | P | R | Recall by box height (< 40 / 40–100 / > 100 px) | Both wickets found | Edge error median / p90 |
|---|---|---|---|---|---|
| Test | 0.58 | 0.58 | 5/13, 21/26, 16/34 | 7 of 18 frames | 1.4 / 7.1 px |
| Val | 0.58 | 0.57 | 8/21, 21/31, 17/29 | 11 of 27 frames | 1.3 / 6.2 px |

**Calibration from two stumps boxes** (`ml/calibration/stumps_landmarks.py`).
- Each box gives 4 points: the outer stump edges at ground and bail-top height.
- Both wickets give 8 points in two parallel planes, which is enough for the approximate-focal pose.
- Broadcast delivery views are long telephoto shots (focal length 2–15× the image width). The old single initial focal guess converged to a mirrored pose on some cameras, so the stumps path now starts from a log-spaced grid of guesses. `estimate_pose` gained an optional `focal_guess_px`; manual calibration is unchanged.

**Synthetic benchmark** (`scripts/evaluate_stumps_calibration.py`, `stumps_calibration_synth.json`).
- Setup: 300 random behind-the-bowler cameras, 40–100 m back, 6–20 m high, far wicket 80–130 px tall.
- Boxes are exact projections, with Gaussian noise added to each edge.

| Box noise | Failures | Focal error median / p90 | Ground error near striker median / p90 | Stump-plane error median / p90 |
|---|---|---|---|---|
| 0 px | 4 | 0.0 / 0.1% | 1.5 / 1.7 cm | 0.3 / 0.4 cm |
| 0.5 px | 3 | 2.5 / 7.0% | 3.5 / 9.2 cm | 0.4 / 0.8 cm |
| 1 px | 8 | 4.9 / 15% | 6.6 / 18 cm | 0.8 / 1.4 cm |
| 2 px | 37 | 9.2 / 28% | 13 / 32 cm | 1.5 / 2.7 cm |

The 1.5 cm floor at zero noise comes from box edges not lining up exactly with the stump corners (perspective and stump thickness). The zero-noise failures are low cameras refused by the pair rule below.

**Real clips from other matches** (`calibrate_cameras.py auto`; 40 clips, every 6th of `hf-cricket-delivery-videos`, 4 series). The first version accepted 6 calibrations, and **2 were false**:
- two stacked boxes on one wicket (the bails half and the base half);
- a fielder's legs plus a far object, with no pitch in view, which still had a reprojection RMS of 1.8 px.

**Eight points fit almost any camera, so a low RMS proves nothing.** Two checks were added:
- **Pair geometry.** The near wicket must be at least 5% taller, with its base at least 1.5 near-wicket heights below the far base. Synthetic broadcast cameras show this refuses about 13% of genuine pairs from low cameras.
- **Camera plausibility.** The camera must be at least 45 m from pitch centre along the pitch line, 3–60 m high, and within 15 m of the pitch axis.

With both checks:
- **3 of 40 clips calibrate** (video061, video085, video223). In all three, the projected creases and return creases land on the painted lines at both ends.
- 37 are refused: the near wicket is not detected (usually hidden behind the umpire or bowler), no consistent pair is found, or every pair fails the checks.
- There is no independent ground-truth camera, so this is a visual check, not an accuracy measurement.

**Conclusion:**
- Automatic calibration works when both wickets are cleanly visible, and it now refuses rather than guesses otherwise.
- Its coverage (3 of 40) is too low to replace manual clicks.
- More stumps training data (better recall and edges) and crease detection (to calibrate from one wicket) are the next steps.

**Crease registration from one wicket (experimental, off by default).**

`ml/calibration/pitch_registration.py` fits the full pitch model (stumps plus every crease at both ends) to a white-line mask. It works like this:
- A stumps box is axis-aligned, so it carries no perspective lean. One wicket therefore leaves both the focal length and the camera height free.
- For each (focal, height) cell, the level camera is fitted to the stumps corners, and its pose is then refined against the creases with a chamfer distance.
- The best candidates are re-ranked with a reverse (recall) term. Without that term, a far-away, long-lens camera can collapse both ends of the pitch onto one set of creases and still score well.
- The final step refines focal length and pose together.

Synthetic frames (`scripts/evaluate_pitch_registration.py`, `pitch_registration_synth.json`):
- Setup: 40 random cameras, the near stumps hidden by an "umpire", and 1 px of noise on the far box edges.

| | Cameras | Ground error near striker, median / p90 / max | Focal error median |
|---|---|---|---|
| All | 40 (0 failures) | 14 / 48 / 54 cm | 16% |
| Accepted (crease inlier ratio ≥ 0.85, ≥ 40 bowler-end inliers) | 6 | 7.3 / 10.5 / 12 cm | 8% |

**Real footage: negative result.** Even the visually correct two-wicket cameras (video061, video085, video223) matched only 13–33% of crease points within 2 px.
- On these broadcasts the creases are often only 5–15 grey levels brighter than the pitch, about the same as the pitch texture. Players and worn ends hide much of the rest.
- Three masks were tried: round top-hat, pitch-area statistics, and directional top-hat with shape filtering. None separated the creases reliably.
- The registration is therefore **not used** by `calibrate_cameras.py auto` or the API unless `--creases` is passed.
- A learned crease or keypoint detector would be needed to make it work on real footage.

**In the web console:** *Auto-detect* on the calibration panel (`POST /api/calibration/auto`) places the standard landmarks on the frame.
- With a two-wicket calibration, every landmark that projects inside the frame is placed.
- Otherwise the four stumps corners are placed.
- Nothing is stored. The user checks and corrects the points, then computes the calibration as before.

## 4g. Broadcast pan/tilt/zoom compensation

Broadcast delivery cameras zoom in (often 1.2-2x, sometimes far more) and follow the play while the ball is in flight. Previously the calibration made on one frame was applied to every frame, which is wrong as soon as the camera moves. `ml/video/ptz.py` now relates each frame of the shot to the calibration frame by a homography (exact for rotation and zoom about a fixed optical centre). The homographies come from ORB features with a MAGSAC fit, chained frame to frame. Static on-screen graphics are masked out, and moving players are rejected as outliers. Ball positions and apparent radii are mapped into the calibration frame before the 3D fit. Observations whose frame cannot be chained to it (a cut, or too few matches) are dropped and reported.

It is used by `POST /api/lbw/analyze` (when the calibration records `frame_index`, which the web console now sends) and by `run_drs_analysis.py --camera-frame`.

**Real footage** (`scripts/evaluate_ptz.py`, 34 clips from `hf-cricket-delivery-videos`, 45-frame delivery window cut at the first shot cut, reference frame in the middle). The wickets do not move, so independently detected stumps bases should land on one reference pixel. Residuals include stumps-detector noise (about 1-2 px). Report: `outputs/reports/ptz_stabilisation_real.json`.

| | Clips | Residual median (clip median) | Residual p90 (clip median) |
|---|---|---|---|
| Raw image positions | 26 measurable | 11.1 px | 30.6 px |
| Mapped into the reference frame | 28 | **3.2 px** | **8.7 px** |

- 6 clips had too few stumps detections to evaluate. 2 clips had no raw value: camera motion spread every landmark too widely to group.
- The raw figure understates the raw error, because a landmark smeared by camera motion can split into several groups.
- Failures remain. On video013, video028 and video088 the stabilised median stays at 12-14 px (fast pans, large zoom, few static features).

**Effect on LBW projection** (synthetic, `scripts/evaluate_zoom_effect.py`, 120 held-out cricket-synth clips, front-on rig). The exact camera is zoomed about its principal point by a constant rate per frame, relative to a calibration frame in the middle of the flight. *Compensated* adds 1.5 px Gaussian noise in place of the measured stabilisation residual. Report: `outputs/reports/zoom_effect_synth.json`.

| Case | Stump-plane error median / p90 | Inconclusive | Definite | Confident errors | 90% interval coverage |
|---|---|---|---|---|---|
| No zoom (baseline) | 5.4 / 18 cm | 39% | 73 | 0 | 0.87 |
| 0.5%/frame, ignored | 8.5 / 32 cm | 46% | 65 | 0 | 0.66 |
| 0.5%/frame, compensated | 7.5 / 26 cm | 44% | 67 | 0 | 0.88 |
| 1%/frame, ignored | 17.8 / 57 cm | 56% | 53 | 1 false miss | **0.42** |
| 1%/frame, compensated | 8.0 / 33 cm | 46% | 65 | 0 | 0.91 |

An ignored 1%/frame zoom triples the error and halves the coverage of the uncertainty interval: the interval becomes overconfident and a confident wrong verdict appears. With compensation the intervals stay calibrated. The remaining loss against the baseline comes from the extra 1.5 px residual noise.

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

## 7. Live capture and synchronisation (synthetic sources, this PC)

**Capture throughput.**
- Setup: synthetic frames with sensor-like noise, 5 s after a 1 s warm-up, on a Ryzen 7 7700 + RTX 3080 Ti.
- Raw output: `outputs/reports/capture_benchmark.json` (`scripts/capture_cameras.py benchmark`).

| Encoder / mode | Config | Frames dropped | Encode ms/frame |
|---|---|---|---|
| NVENC, threads | 720p at 50/100/120 fps | 0 | 1.8–2.2 |
| NVENC, threads | 1080p at 50/100/120 fps | 0 | 3.1–3.5 |
| NVENC, process per camera | 2 × 1080p100 | 0 | 4.1–4.6 |
| NVENC, process per camera | 3 × 1080p100 | 0 queue drops, near the limit | 9.7 |
| libx264 | 1080p100, 1 camera | 0 | 8.6 |
| libx264 | 2 × 1080p100 | 304 | 15.3 |

- **Bitrate:** about 35 MB/s per 1080p100 camera on noisy content.
- **Timing:** exported clips reproduce the recorded host arrival times to within 0.006 ms.
- **Webcam:** one physical webcam declared 30 fps but delivered about 15 fps. The mismatch is detected and flagged.
- **Late arrivals:** in an end-to-end run through the API (two synthetic cameras, 50 + 100 fps, alongside the web server), about 1–2% of frames arrived more than 1.5 intervals late because of host scheduling stalls.
  - These frames were not lost: the device frame counter was contiguous, and no frames were missing from the clip.
  - They are now reported as `late_arrivals` instead of drops.
  - Their arrival times are still delayed, which is one more reason software sync is approximate.

**Microphone capture on the shared clock.**
- Measured on this PC: 4 s recordings, export window fitted with the nominal sample rate.

| Device / host API | Timestamp basis | Block-time residual | Offset uncertainty |
|---|---|---|---|
| UGREEN camera mic, WASAPI, 48 kHz | PortAudio ADC time | 0.64 ms | 0.11 ms |
| L30 PRO mic, MME, 44.1 kHz | callback arrival | 11.2 ms | 1.7 ms |

- **Synthetic clicks** with 2 ms reported-timestamp jitter: every click is placed within 1.5 ms of its true host time after export.
- **What this does not include:** the microphone's own latency and the sound's travel time (about 2.9 ms per metre). Using WASAPI devices is recommended.

**Sync sensitivity.**
- Setup: synthetic end-on + side-on 1080p rig, ball at about 37 m/s, triangulated with a given sync error, flight-model fit, projected to the stump plane. Script: `scripts/evaluate_sync_sensitivity.py`; output: `sync_sensitivity.json`.
- 3D position error grows by about 3.54 cm per ms of sync error, regardless of frame rate.
- Stump-plane error at 100 fps with 1 px noise (median of 20 runs):

| Sync error | 0 ms | 1 ms | 2 ms | 5 ms | 10 ms | 20 ms |
|---|---|---|---|---|---|---|
| Stump-plane error | 1.1 cm | 1.8 cm | 2.5 cm | 4.7 cm | 8.2 cm | 14.7 cm |

- Reprojection error stays near 1–2.5 px at these errors, so **it cannot detect sync error**.

**Audio sync.** GCC-PHAT recovers a synthetic 123.375 ms offset to within 0.4 µs. File-based offsets come back within 0.2 ms, and pure noise returns `LOW_CONFIDENCE`. Not tested on real stadium audio.

## 8. Run-out, stumping, no-ball and boundary engines

- Covered by deterministic tests on synthetic geometry and footage: crease sign conventions at both ends, metric distance, near-line ambiguity, wicket-break detection on encoded video, timing-window logic, Law 30.1.2, Law 38/39 exceptions, Law 19 six/four, the 2026 airborne-fielder rule, and Law 21.5 no-ball geometry (front foot behind or on the line, the near-line inconclusive case, back foot vs return crease seen from behind the bowler).
- No labelled real footage was available, so **real-world accuracy is unknown**.
- The engines' value is exact geometry and timing given correct marks. Their failure mode is wrong marks.

## 9. Crease mark proposals (pose model, unlabelled broadcast clips)

**Model and method.** `yolo11s-pose.pt` (COCO keypoints, AGPL-3.0, not cricket-trained) at 1280 px.
- The batter is the person nearest the crease line or inside the operator's region; there is a continuity term between frames.
- The proposals are the ankles, plus a sole estimate on the shin line.
- Each ankle also gets a heuristic grounded hint: the lower foot with less than 0.012 body heights of vertical motion per frame counts as grounded. The hint needs both neighbouring frames, otherwise it is `null`.
- No bat-tip proposals: there is no bat model.

**Data.** Script: `scripts/evaluate_crease_proposals.py`; report: `outputs/reports/crease_proposals_eval.json`.
- `hf-cricket-delivery-videos`, every 5th clip (52 clips), first 75 frames. Broadcast, 1920×1080 at 25 fps, behind-the-bowler views.
- The region was seeded around the striker's stumps by the stumps detector, in place of the operator. 40 clips had a stumps box in the first 10 frames; 12 were skipped.
- There are **no labels**, so none of these numbers is accuracy.

| Measure | Value |
|---|---|
| Frames with a batter selected (40 clips × 75 frames) | 39.6% |
| Visual check of the selected person at frame 25 (40 clips) | Batter 17, wrong person 6 (wicket-keeper 5, umpire 1), nobody 17 |
| Ankle keypoint confidence (n = 2,327) | p10 0.94, median 0.98, p90 0.99 |
| Grounded hints | Grounded 962, not grounded 203, no hint 1,162 |
| Inference speed, RTX 3080 Ti, FP16, batch 16 | Median 115 fps (one clip at 16 fps, including CUDA warm-up) |
| Inference speed, CPU | 5.5 fps |

**Findings.**
- **"Nobody selected"** has three causes: the broadcast cuts to close-ups and wide shots, the stumps detector placed the region wrongly (for example on the umpire's pocket), or the batter left the region after the shot. In an operator workflow the region or crease is drawn on the replay itself.
- **Wicket-keeper confusion.** When only a region is given, the keeper standing up to the stumps is chosen instead of the batter in 5 of 22 clips where someone was selected. Giving the crease line resolves most of these cases, because the keeper's feet are about 1.2 m or more behind it. The response warns about this.
- **No-region fallback removed.** An earlier version picked the largest person when neither a crease nor a region was given. On 16 clips without stumps it chose the umpire, non-striker, bowler or a TV-graphic face in at least 7, so the fallback was removed. The API now requires a crease or a region.
- **Keypoint confidence is not accuracy.** High ankle confidence says nothing about pixel accuracy under pads and motion blur. With no labels, localisation error is **not measured**.
- **Camera motion.** The grounded hint assumes a static camera, which broadcast pans violate. It is a prompt for the operator, never an observation.

## 10. Multi-view LBW (synthetic)

- **Code:** `ml/lbw/multiview.py` (`POST /api/lbw/multiview`), script `scripts/evaluate_multiview_lbw.py`, output `outputs/reports/multiview_lbw_synth.json`.
- **Setup:**
  - Two exactly calibrated 1920×1080 cameras: end-on at 50 fps (4200 px focal) and side-on at 100 fps (2600 px focal) with a 3 ms phase offset, so exposures never coincide and the side-on view is interpolated to the end-on instants.
  - 60 random deliveries: pitching 3–7 m from the stumps, line ±0.25 m, stump-plane x ±0.35 m, 30–40 m/s. The truth model includes aerodynamic drag.
  - 1 px image noise; impact at 1.2 m in front of the stumps.
  - Camera B really exposes `error` ms later than its timestamps claim.

Current defaults (swing and lift terms, covariance inflation ×8; chosen in §13 because the earlier drag-only model made confident errors on richer physics). The earlier defaults are in brackets.

| Scenario | Actual / declared sync error | Definite (correct) | Confident errors | Inconclusive | Hits called hit | Stump-plane error median / p90 |
|---|---|---|---|---|---|---|
| hardware sync | 0 / 0 ms | 47 (47) [54 (54)] | 0 [0] | 22% [10%] | 13 / 22 [18 / 22] | 1.1 / 1.9 cm [1.1 / 2.4] |
| software sync | 2 / 2 ms | 0 | 0 | 100% | 0 / 22 | 2.1 / 3.7 cm |
| software sync | 20 / 20 ms | 0 | 0 | 100% | 0 / 22 | 59 / 196 cm |
| hardware claimed, error undeclared | 2 / 0 ms | 48 (48) [54 (54)] | 0 [0] | 20% [10%] | 14 / 22 [18 / 22] | 2.1 / 3.7 cm |

**Reading the table:**
- **With synchronised cameras, two views remove most of the monocular depth problem.** 47 of 60 deliveries got a definite verdict, all correct, and 13 of 22 hits were called as hits. That compares with about 1 in 20 hits for the single-camera engine (§2). This generator's deliveries have drag only and no lateral forces, so the wider intervals of the current defaults cost decisiveness here; on richer physics they are what prevents confident errors (§13).
- **Software sync (any declared uncertainty) never gives a definite verdict.** At about 26–35 m/s near the pad, 2 ms is already 5–7 cm, more than the 3 cm tolerance (`configs/lbw.yaml` › `multiview.max_sync_position_error_m`). The sync error is also propagated into the projection's covariance by re-fitting with each camera's offset shifted by its uncertainty.
- **20 ms of sync error (a typical USB-webcam rig) wrecks the geometry.** The median stump-plane error is 61 cm, much worse than the 15 cm of the matched-frame study in §7, because interpolating the side-on track at the wrong instants also corrupts the association. Such rigs can only support replay review, not ball tracking.
- **An undeclared 2 ms error is not detected.** Verdicts stayed correct in this set because a 2 cm stump-plane shift rarely crosses the decision boundary, but nothing in the reconstruction (including reprojection error) reveals the error. The declared sync uncertainty must come from the hardware, not be assumed.
- **Synthetic only:** exact calibration, a known detector-free track, and no motion blur or occlusion. Real-world accuracy is unknown.

## 11. TensorRT and pipeline throughput

Measured on 2026-10-18 on an RTX 3080 Ti with TensorRT 10.16.1, torch 2.11 + CUDA 12.8 and Ultralytics 8.4.175, on native Windows. Raw outputs are in `outputs/reports/trt_export_cricket_ball_best_1280.json` and `inference_benchmark_trt_e2e.json`.

**Engine.**
- Source: `cricket_ball_best.pt` (v3), exported to an FP16 dynamic engine.
- Input profile: up to batch 8, letterboxed inputs up to 736×1280, which covers 16:9 frames at imgsz 1280 (`scripts/export_tensorrt.py`).
- Build: about 2 minutes with a 4 GB builder workspace.
- Build attempts that did not ship:
  - A square 1280×1280 profile at max batch 8 failed to build with a 4 GB workspace (no tactic for the attention block).
  - With an 8 GB workspace that profile built, but needed a 6.7 GB execution context.
  - The 16:9 profile's context is 4.3 GB.
- Engine builds are not bit-reproducible: two builds of the same profile differed by one low-score detection.

**Agreement with PyTorch FP16** on the 298 independent broadcast test frames (`broadcast_seg_test`, 213 labelled balls, imgsz 1280, score ≥ 0.25):

| | Precision | Recall | AP (centre match) |
|---|---|---|---|
| PyTorch FP16 | 0.634 | 0.512 | 0.532 |
| TensorRT FP16 | 0.634 | 0.512 | 0.531 |

- **Detection matching:** 171 of 172 PyTorch detections are reproduced (IoU ≥ 0.5), with one engine-only detection.
- **Median centre difference:** 0.0 px; the earlier square profile gave 0.26 px.
- **Validation gate:** the export script writes a sidecar recording these numbers. The detector uses the engine only if all of the following hold:
  - the sidecar says validation passed (agreement ≥ 0.97, extra detections ≤ 3%, median centre difference ≤ 1 px, AP drop ≤ 0.01);
  - the `.pt` and engine hashes, the TensorRT version and the GPU still match.

**Detector throughput** (1080p broadcast clip, 400 pre-decoded frames, imgsz 1280; median of batch latencies):

| Runtime | Batch | fps | Batch latency median / p95 (ms) | Device memory (MB) |
|---|---|---|---|---|
| PyTorch FP16 | 1 | 87 | 10.0 / 16.0 | 201 |
| TensorRT FP16 | 1 | 158 | 6.2 / 7.7 | 4943 |
| PyTorch FP16 | 8 | 146 | 53.8 / 69.8 | 774 |
| TensorRT FP16 | 8 | 191 | 40.6 / 51.3 | 5046 |
| PyTorch FP16 | 16 | 152 | 104 / 140 | 1489 |

- Device memory is the device-level increase while the model is loaded and running; TensorRT allocates outside PyTorch's allocator.
- PyTorch results were noisy across runs: an earlier run on the same clip gave about 92 fps at every batch size, with one 1.4 s stall. The TensorRT rate was stable at 188–191 fps.
- Requests above batch 8 are clamped to the engine's build limit.

**End to end.**
- *Decode + detect, 1000 frames of the 1080p clip, batch 8* (decode alone runs at 650–820 fps):
  - PyTorch: 123–129 fps; TensorRT: 131–154 fps.
  - Preprocessing (letterbox on the CPU) and postprocessing, not the network, bound this.
- *Full pipeline* (detection, tracking, trajectory fit, no rendering; 526-frame 1080p clip):
  - Run time: 11.1 s with PyTorch, 6.8 s with TensorRT. This includes model loading.
  - The primary ball track is identical: the same 83 observed frames, maximum centre difference 0.75 px, reliable in both.

**Decode/inference overlap.** A decoder thread feeding a bounded queue (4 batches) was compared with serial decoding, alternating runs:

| Runtime | Serial (fps) | Overlapped (fps) |
|---|---|---|
| PyTorch | 122.6–125.1 | 124.0–131.3 |
| TensorRT | 137.5–149.4 | 130.6–154.0 |

The effect ranges from −5% to +7%, within run-to-run noise. Decoding is already about 5 times faster than detection, so there is little to hide. **The production pipeline was not changed.**

**Decision.** The engine is opt-in: `PipelineConfig.use_tensorrt=True`. It gives about 1.3× detector throughput and 1.6× lower batch-1 latency, with the same detections, at a cost of about 4.3 GB more VRAM. On a 12 GB card that is acceptable for offline analysis, but it competes with NVENC capture and a second model during a match.

Frames whose letterboxed shape is larger than the engine profile (4:3, portrait) are sent to the `.pt` model, never silently resized. A test covers this.

## 12. Audio edge transients (real broadcast audio, unlabelled)

`scripts/evaluate_audio_transients.py` runs the transient detector on 60 `hf-cricket-delivery-videos` clips (44.1 kHz AAC, about 6 s each). Settings: high-pass 2 kHz; at least 15 dB above the noise floor; rise time ≤ 4 ms. Raw output: `audio_transients_broadcast.json`.

| Measure | Value |
|---|---|
| Clips with audio | 60 / 60 |
| Clips with at least one transient | 25 |
| Transients per clip, median / p90 | 0 / 5.2 |
| Runtime | 4.7 s for all 60 clips (CPU) |

**A bug found on real audio and fixed.**
- The first version measured rise time from the *first* 10% crossing in the 10 ms before the peak. An earlier noise blip therefore inflated the rise time, and real impacts were rejected.
- Example: `video001` has a 35 dB, 3 ms impact at 969 ms that was missed.
- Rise time is now measured back from the peak, and a regression test covers this. The earlier figure of 14 clips with a transient came from the buggy version.

**Qualitative check** (not a measurement):
- For 8 clips with 1–2 transients, the frame nearest the first transient was inspected.
- In 7 the batter is at or just after the stroke, consistent with bat-ball contact within a frame or two.
- In 1 (`video014`) the frame shows a fielder diving, a ground-contact sound and one of the documented alternative explanations.
- Clips with many transients (up to about 5) carry crowd or stadium noise.

**What this cannot tell:**
- **No contact labels exist**, so edge precision and recall, especially for faint edges, are **unknown**.
- Broadcast audio is mixed and compressed; a stump microphone recorded at ≥ 48 kHz would be the real input.

**Synthetic tests:**
- A 7 kHz click is located within 1 ms. A slow 220 Hz sound is rejected by the rise-time test. A noise blip 6.5 ms before an onset no longer inflates the rise time.
- Through an AAC-encoded MP4 the click is found 2.5 ms early (encoder priming), one reason the API warns that the audio/video offset is unknown.

## 13. Physics refinement (synthetic)

**Question:** which extra flight-model terms help? The terms tested are per-delivery drag, a separate post-bounce lateral term, and swing/lift in the 3D (multi-view) fit. A term is kept only if it improves held-out results without adding confident errors.

**Data and protocol:**
- Only the `front_on_stump` rig of cricket-synth has rendered labels (405 clips; the upstream repository has 410 label files). The manifests also describe 300 clips for other rigs, but those were never rendered.
- For the multi-view experiment, the same simulated 3D tracks are re-projected through manifest `side_on` cameras (exact projections, 1 px noise).
- **Tuning** used clips 0–119. **Final numbers** use clips 240–404, which no earlier tuning touched.
- True per-delivery drag coefficients come from the manifests: Cd 0.38–0.52, median 0.45, against the fixed 0.40 the model assumes.
- Script: `scripts/evaluate_physics_refinement.py`; output: `outputs/reports/physics_refinement_synth.json`.
- **Prior choice:** the Cd prior N(0.40, 0.10) is deliberately not centred on the simulator's median. Published cricket-ball drag data show considerable scatter, and the prior should not encode this simulator.

**1. Single camera (LBW benchmark protocol of §2), held out, n = 165:**

| Variant | Projected | Stump-plane median / p90 | 90% coverage | Inconclusive | Definite (all correct) | Hits confirmed |
|---|---|---|---|---|---|---|
| **fixed drag (default, unchanged)** | 152 | 7.0 / 36.6 cm | 0.82 | 0.41 | 97 | 3 / 22 |
| fitted drag, Cd ~ N(0.40, 0.10) | 153 | 6.8 / 36.9 cm | 0.81 | 0.42 | 95 | 3 / 22 |
| separate post-bounce lateral term | 162 | 6.2 / 30.6 cm | 0.85 | 0.45 | 90 | 2 / 22 |
| both | 162 | 6.2 / 30.5 cm | 0.85 | 0.47 | 87 | 1 / 22 |

- **Drag is not identifiable from one 60 fps camera.** The posterior sd of Cd is 0.099 against the prior's 0.10 (ratio about 0.99). The fitted value stays at the prior (bias −0.045 against the true Cd) and only adds uncertainty.
- **The separate post-bounce term** lowers the error and raises coverage, but gives fewer definite verdicts and fewer confirmed hits. **Neither is enabled by default.** Both remain available as options: `drag_cd_prior` and `swing_post_prior_sigma` in `fit_to_pixels`.
- **Coverage below nominal:** this held-out set's 90%-interval coverage (0.82) is lower than the 0.87 measured on clips 120–239.

**2. 3D fit on ground-truth positions + 1 cm noise (multi-view quality), held out, n = 165, stump-plane error median / p90:**

| drag only (previous multi-view model) | + swing | + swing + lift | + swing + lift + fitted drag |
|---|---|---|---|
| 7.7 / 32.3 cm | 5.7 / 23.9 cm | **5.4 / 17.7 cm** | 5.5 / 20.3 cm |

- With 3D data, drag *is* identifiable: the posterior/prior sd ratio is 0.28–0.37 (exploratory run, tuning clips 0–59), with a Cd error of about 0.03.
- Fitting drag still does not improve the stump-plane prediction. Drag mainly changes timing along the pitch, not where the ball crosses the stump plane.

**3. Multi-view LBW (front-on + side-on, hardware sync, 60 fps), held out, n = 165:**

| Variant | Definite (correct) | Confident errors | 90% coverage | Inconclusive | Hits confirmed | Stump-plane median / p90 |
|---|---|---|---|---|---|---|
| drag only, inflation ×2 (previous default) | 131 (129) | **2 false misses** | 0.26 | 0.21 | 13 / 22 | 6.7 / 28.8 cm |
| + swing + lift, ×2 | 137 (136) | 1 false hit | 0.52 | 0.17 | 16 / 22 | 5.2 / 19.9 cm |
| **+ swing + lift, ×8 (new default)** | 118 (118) | **0** | 0.84 | 0.29 | 10 / 22 | 5.2 / 19.9 cm |

- **The confident errors were 8–11 cm misses with intervals only about 4 cm wide.** The balls were near bail height and bounced off a friction-and-spin pitch model. The drag-only fit's uncertainty did not cover model misspecification: its "90%" intervals contained the truth 26% of the time.
- **Choosing the inflation:** it was set on clips 0–119 to the smallest value reaching about 90% coverage.

  | Inflation | 2 | 4 | 8 | 16 |
  |---|---|---|---|---|
  | Coverage | 0.62 | 0.78 | 0.90 | 0.97 |
  | Definite verdicts (all correct) | 105 | 102 | 96 | 86 |

- **Cost of the change:** fewer definite verdicts and confirmed hits, here (131 → 118, 13 → 10 of 22) and on §10's drag-only deliveries (54 → 47, 18 → 13 of 22).
- **Why it is accepted anyway:** a review aid that is wrong with high confidence is worse than one that defers.
- **Remaining gap:** held-out coverage is 0.84, slightly below nominal, so a residual risk of overconfidence remains.

**Changes made:**
- `MultiViewConfig` now defaults to swing (prior σ 2 m/s²), lift (σ 3 m/s²) and covariance inflation ×8 (`configs/lbw.yaml` › `multiview`).
- `fit_to_points` gained optional `swing_prior_sigma`, `lift_prior_sigma` and `drag_cd_prior`, all requiring per-point sigmas.
- `fit_to_pixels` gained optional `drag_cd_prior` and `swing_post_prior_sigma`, both off by default.
- The single-camera defaults are unchanged.

**Synthetic only.** These results say which model terms help against one simulator's physics with exact cameras. They are not evidence of real-world accuracy.

## 14. LBW end to end through the running app (synthetic deliveries)

`scripts/run_lbw_scenarios.py [--multiview]` renders six deliveries with known outcomes and puts each through the real workflow against the running API. The steps are:
1. upload and run a tracking job (motion detector);
2. calibrate with exact intrinsics;
3. run LBW (single camera, or `/api/lbw/multiview` with an added side-on 100 fps camera and hardware sync asserted);
4. open an LBW review under ICC T20I 2025 and compute the recommendation.

The flight includes realistic air drag (`make_delivery(drag_c=DEFAULT_DRAG_C)`). Raw output: `lbw_scenarios_e2e*.json`.

| Scenario (truth) | Single camera: verdict, recommendation | Two cameras: stump-plane error, verdict, recommendation |
|---|---|---|
| Hitting middle (centre in zone) | INCONCLUSIVE, refer to umpire | 2.2 cm, INCONCLUSIVE (P(centre in zone) 0.92 < 0.95), refer |
| Missing off, 32 cm wide | PROJECTED_TO_MISS, OUT overturned | 2.8 cm, PROJECTED_TO_MISS, OUT overturned |
| Missing leg | PROJECTED_TO_MISS, OUT overturned | 1.3 cm, PROJECTED_TO_MISS, OUT overturned |
| Going over the top | INCONCLUSIVE, refer | 2.4 cm, PROJECTED_TO_MISS, OUT overturned |
| Clipping off stump (part in zone, centre out) | INCONCLUSIVE, NOT OUT stands | 2.0 cm, INCONCLUSIVE, NOT OUT stands |
| Pitched outside leg (would hit) | OUT overturned (Law 36.1.2) | 1.6 cm, OUT overturned (Law 36.1.2) |

- **No recommendation contradicted the truth** with either set-up.
- **Single camera:** heights come out about 20 cm high. This is a consistent bias from the monocular depth assumptions (bounce speed retention prior), so uncertain cases are referred to the umpire.
- **Two cameras:** stump-plane errors are 1.3–2.8 cm.

**Defects found and fixed by this test:**
1. **Grayscale motion detection** missed a red ball on green grass, which have nearly equal luminance. The detector now differences in colour.
2. **Ghost candidates.** Three-frame differencing left a ghost where a slow ball had been. Ghosts are now rejected against a 15-frame median background.
3. **Duplicate tracker hypotheses** followed the same ball near the bounce and took the post-bounce detections, so the ball track ended at the bounce. A recorded fixture now tests this. Re-tracking the 5 real broadcast clips gives identical results (3/5 reliable).
4. **The protocol layer** let an INCONCLUSIVE *would-hit* verdict override a certain "pitched outside leg". It now only blocks decisions that depend on the wicket projection.
5. **The synthetic generator had no air drag**, while the engine assumes a real ball. Drag is now optional in the generator and used by this test.

These are synthetic scenes. Real-world LBW accuracy remains unknown.

## 6. Not measured (no ground truth available)

- Detection metrics on real cricket footage.
- Tracking continuity and fragmentation against labelled real sequences.
- 3D error on real deliveries.
- LBW accuracy against reference decisions.
- Real multi-camera synchronisation sensitivity.
- Contact detection.
- Ankle / ground-contact localisation error of the crease mark proposals (§9).
