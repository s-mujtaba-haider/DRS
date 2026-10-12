# Limitations

These limitations are measured or known. They are not hypothetical.

## Not an umpiring system
- No result is an official decision or a certified DRS verdict.
- The **real-world accuracy of every decision output is unknown.** No dataset with independent LBW or ball-tracking ground truth on real footage was available.

## Detection (the weakest component)
- **Broadcast detection is still the weakest link.** The current v3 detector reaches recall 0.51 and precision 0.63 (AP 0.53) on independent broadcast frames (v2: 0.53 / 0.62 / 0.49; v1: 0.37 / 0.64 / 0.38), so about half of tiny broadcast balls are missed. Its training labels include self-generated (track-verified) pseudo-labels, which favour balls earlier models could already see. The second round of pseudo-labels is noticeably noisier (about 70% clearly a ball in a 48-crop sample), so further self-training rounds are not expected to help without stricter filtering or human labels.
- With it, 3 of 5 real broadcast clips produce a consistent track that is visually the ball (v2: 2 of 5). All three follow the ball **after** the shot (in the air or across the outfield), not the delivery itself, which is what LBW analysis needs.
- Generic COCO plus motion failed on all 5 clips (mean innovation 47–616 px).
- **The motion heuristic** is confused by players, the bat, fielders and camera motion. It works on simple synthetic scenes.
- **Broadcast footage** cuts between cameras, zooms and overlays graphics:
  - Cuts are detected heuristically and tracks are never joined across them.
  - Replays, slow motion and picture-in-picture are not specially handled.

## Tracking
- The reliability flag (≥ 8 observations, ≥ 50% observed, mean innovation ≤ 12 px) is a *necessary* consistency check. It is not proof that a track is the ball.
- Tracking accuracy against labelled real footage has **not been measured**, because no labelled real sequences exist yet.

## Timing
- 25–50 fps broadcast video means the ball moves 0.8–1.5 m between frames at 140 km/h.
- Impact time, bounce time and any contact event are therefore quantised coarsely.
- Contact analysis returns `INCONCLUSIVE` below 100 fps by design.

## Single-camera 3D
- Depth comes from physical assumptions, not from a second view:
  - gravity;
  - fixed aerodynamic drag (Cd ≈ 0.4);
  - a fitted lift term with a prior;
  - the ball resting on the ground at the bounce;
  - a prior on speed retention through the bounce;
  - the apparent ball size, used only for learned-detector boxes.
- Post-bounce depth along a behind-the-bowler camera's line of sight is poorly observable, and the covariance shows it.
- **Approximate calibration** (focal length estimated from landmarks) assumes the principal point at the image centre, square pixels and no lens distortion. With unvalidated calibration the engine will not give a definite verdict by default.
- **On the cricket-synth benchmark** (known camera, 1 px noise, 120 held-out clips; see `EVALUATION.md`):
  - Median stump-plane error is about 6 cm, with p90 about 26 cm and occasional outliers of metres.
  - About 40% of deliveries come out `INCONCLUSIVE`. There are no confident errors.
  - **Only 1 of 20 deliveries that would hit the stumps received a definite `PROJECTED_TO_HIT`.** The engine remains conservative on hits.
- These are synthetic results with an exactly known camera. Real calibration error, detector noise and motion blur will make results worse.

## Uncertainty calibration
- The covariance inflation factor (×2, with the swing model) was tuned to reach about 90% interval coverage on one set of 120 synthetic clips. On a separate 120 clips coverage was 0.87.
- It has not been validated on real deliveries.

## Rules layer
- The rules layer implements the Law 36 conditions that geometry can inform.
- Other conditions stay `UNKNOWN` unless the user supplies them:
  - no ball;
  - bat first, including the simultaneous bat-and-pad rule;
  - interception by the person;
  - genuine attempt to play.
- The off side is taken from user-entered handedness. Switch hits are not handled.
- Competition-specific DRS protocol (umpire's call thresholds, review counts) is **not** implemented as a decision. `p_centre_in_zone` is reported for reference only.

## Not implemented
Run-out, stumping and boundary analysis; synchronised multi-camera ingestion; audio (snicko); thermal; bat detection; pad and body segmentation; automatic crease detection.

## Automatic calibration from stumps (experimental)
- The stumps detector was trained on **one match**. It finds both wickets in only 7 of 18 held-out frames that show both, and its box edges are off by more than 7 px in 10% of cases.
- It needs both wickets. In delivery views the near wicket is often hidden by the umpire or bowler, so on 40 clips from other matches only 3 produced a calibration (all 3 visually correct); 37 were refused.
- A low reprojection error is **not** evidence of a correct calibration: 8 points fit almost any camera. False pairs are rejected by geometry rules (near wicket taller and well below the far one) and a camera-position check. These rules also refuse about 13% of genuine pairs seen from low (< ~6 m) cameras.
- Even with perfect detection, 2 px of box noise gives about 13 cm median and 32 cm p90 ground-position error near the striker (synthetic). The result is a seed for manual correction, never a validated calibration.
- Calibrating from one wicket plus the painted creases works on synthetic frames, but **not on real broadcast frames**: the creases are too faint against the pitch for the current line extraction. It is off by default.
