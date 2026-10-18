# Roadmap

Ordered by expected impact on real-footage usefulness.

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
   - Fit drag per delivery when observations allow it.
   - Add spin and seam lateral forces with priors.
   - Evaluate on cricket-synth's other camera rigs.
7. **Contact analysis.**
   - Bat detection and segmentation, plus contact-region tracking.
   - Synchronised audio only with genuine high-rate audio and validation data.
8. **Run-out and stumping.**
   - Done: crease geometry, wicket-break candidates, the timing engine, Law 38/39 rules and the review UI.
   - Next: a bat-tip and foot keypoint detector to propose marks; break detection that follows camera motion; evaluation on labelled high-frame-rate side-on footage.
9. **Boundary analysis.**
   - Done: rope geometry, the event-sequence Law 19/33 engine and its UI.
   - Next: rope segmentation and fielder pose, so events are proposed rather than hand-marked.
10. **Performance.**
    - TensorRT FP16 engine export for the trained detector.
    - Overlap decoding and inference (decoding already runs at 245–770 fps on the CPU).
    - Use NVDEC where available.
11. **Match-day hardening.**
    - Done: matches, reviews, protocol engine, audit log, evidence packages and live capture.
    - Next:
      - Role-based login for operator vs third umpire.
      - PDF review report.
      - A large replay-monitor view for the third umpire.
      - A rehearsal mode that replays recorded matches through capture.
