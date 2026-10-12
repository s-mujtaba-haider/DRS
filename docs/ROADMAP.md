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
   - Detect creases (white lines on the pitch) so one wicket plus the creases can calibrate when the near wicket is hidden.
   - Load the seeds into the web calibration screen, so manual clicks become corrections rather than the only input.
4. **Delivery segmentation for broadcast.**
   - Filter by shot type (pitch view vs. replay or crowd); `suksham-cricket-frames` provides labels for this.
   - Detect bowl release using the DeepSportradar set, which is non-commercial.
5. **Multi-camera workflow.**
   - Per-camera jobs, then estimate synchronisation offsets (audio cross-correlation or visual events).
   - Associate tracks across cameras and feed them to `/api/reconstruction/triangulate`, then `fit_to_points`.
   - Validate on `kaggle-wide-balls-2view` (front and side views plus calibration material).
6. **Physics refinement.**
   - Fit drag per delivery when observations allow it.
   - Add spin and seam lateral forces with priors.
   - Evaluate on cricket-synth's other camera rigs.
7. **Contact analysis.**
   - Bat detection and segmentation, plus contact-region tracking.
   - Synchronised audio only with genuine high-rate audio and validation data.
8. **Run-out and stumping.** Crease geometry, bat and foot localisation, wicket-break timing, and a frame-accurate review timeline. These need high-frame-rate side-on footage to be meaningful.
9. **Boundary analysis.** Rope detection, field geometry, and fielder-ball-rope interaction evidence.
10. **Performance.**
    - TensorRT FP16 engine export for the trained detector.
    - Overlap decoding and inference (decoding already runs at 245–770 fps on the CPU).
    - Use NVDEC where available.
