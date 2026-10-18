# Roadmap

Ordered by expected impact on real-footage usefulness.

## Blocked on inputs only the project owner can provide
The software for each of these exists. What is missing is data or hardware:
- **Human ball labels on your own broadcast footage** (300–500 frames, split by match). Use the console's *Label frames* page, then `annotate_video.py build`. This is the largest single accuracy gain available (item 1).
- **Labelled real sequences** for tracking evaluation (item 2) and labelled crease and stumps frames for automatic calibration (item 3).
- **Hardware-synchronised high-frame-rate cameras.** Trigger or PTP-locked cameras at 100–240 fps, side-on at both creases and behind the bowler's arm (`CAMERA_HARDWARE.md`). Multi-camera LBW gives definite verdicts only with hardware sync.
- **Reference decisions:** any footage with independently known outcomes (run-out, no-ball and LBW decisions with ground truth), so real-world accuracy can be measured rather than stated as unknown.
- **Acceptance** by the competition organiser before any use in a sanctioned match (`INTERNATIONAL_DRS_REQUIREMENTS.md` sections 8–9).

1. **Broadcast-domain ball detection.** v3 exists (in-domain R 0.97, broadcast R 0.51 / AP 0.53). Two self-training rounds have plateaued; the next gain needs human labels.
   - Label 300–500 small-ball frames from your own broadcast footage (`annotate_video.py` plus UI corrections; split by video).
   - Optionally add licence-checked Roboflow broadcast sets.
   - Retrain and re-evaluate on `broadcast_seg_test`, keeping it test-only.
   - Consider a temporal heatmap model (TrackNet-style, three stacked frames) for blurred, tiny balls.
2. **Tracking evaluation on real labelled sequences.** Measure continuity, fragmentation, false associations and position error with the metrics already used for detection.
3. **Automatic pitch and stump localisation.** A first version exists: a stumps detector (one match of training data) seeds calibration through `calibrate_cameras.py auto` and succeeds on 3 of 40 clips from other matches (`EVALUATION.md` §4f).
   - Add stumps labels from more matches (licence-checked Roboflow sets or own labels); recall and box-edge accuracy are the limits.
   - Crease registration from one wicket is implemented and works on synthetic frames, but hand-crafted line extraction fails on real broadcast creases (`EVALUATION.md` §4f). Next: a learned crease/keypoint detector (heatmaps for crease corners), trained on labelled frames.
   - Done: *Auto-detect* in the web calibration panel places the landmarks for correction.
4. **Delivery segmentation for broadcast.** Done: pan/tilt/zoom compensation within a shot (`EVALUATION.md` §4g). Next: auto-propagate the calibration across a shot so any frame can be the calibration frame, and flag frames where the chain residual is high.
   - Filter by shot type (pitch view vs. replay or crowd); `suksham-cricket-frames` provides labels for this.
   - Detect bowl release using the DeepSportradar set, which is non-commercial.
5. **Multi-camera workflow.**
   - Done: synchronised capture on one host clock, clip export per review, and audio/event sync estimation (`/api/sync/*`).
   - Next: hardware sync (PTP or trigger) for real rigs; run per-camera tracking jobs for a clip group automatically.
   - Associate tracks across cameras and feed them to `/api/reconstruction/triangulate`, then `fit_to_points`.
   - Validate on `kaggle-wide-balls-2view` (front and side views plus calibration material).
6. **Physics refinement.**
   - Done as a study (`EVALUATION.md` section 13):
     - Fitting drag per delivery is not identifiable from one camera, so it stays off.
     - The post-bounce lateral term lowers error but costs definite verdicts, so it is optional.
     - Multi-camera defaults now use swing and lift terms with ×8 inflation and make 0 confident errors.
   - Next: validate interval coverage on real deliveries, once reference data exists.
7. **Contact analysis.**
   - Done:
     - trajectory-deflection candidates;
     - audio transients aligned to frames;
     - stump-microphone capture on the camera clock (about 0.1 ms with WASAPI).
   - Next: bat detection and segmentation for contact-region tracking; validate the audio transients against labelled edges.
8. **Run-out and stumping.**
   - Done:
     - crease geometry, wicket-break candidates (with camera-motion compensation), the timing engine and Law 38/39 rules;
     - pose-based foot proposals and the review UI.
   - Next: a bat-tip detector; evaluation on labelled high-frame-rate side-on footage.
9. **Boundary analysis.**
   - Done: rope geometry, the event-sequence Law 19/33 engine and its UI.
   - Next: rope segmentation and fielder pose, so events are proposed rather than hand-marked.
10. **Performance.**
    - Done: a TensorRT FP16 engine (158 vs 87 fps at batch 1, validated). Overlapping decoding with inference was measured and gave no gain.
    - Next: NVDEC decoding for many-camera live analysis.
11. **Match-day hardening.**
    - Done:
      - matches, reviews, protocol engine, audit log, evidence packages, PDF review reports;
      - operator and umpire PINs, a synchronised replay monitor and second-screen umpire view;
      - live capture with a stump microphone, and rehearsal mode;
      - no-ball checks.
    - Next:
      - TLS and real user accounts for networked deployments;
      - automatic front-foot proposals at the bowler's crease;
      - an operator handbook with timed rehearsal results from a real ground.
