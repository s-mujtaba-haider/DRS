# Roadmap

Ordered by expected impact on real-footage usefulness.

1. **A cricket ball detector.** This is the biggest blocker.
   - Add a Kaggle token and download `kushagra3204/cricket-ball-dataset-for-yolo` (CC0, 1,778 images).
   - Optionally add licence-checked Roboflow exports.
   - Label 300–500 frames from your own target footage with `annotate_video.py` and UI corrections, keeping whole videos as separate splits.
   - Fine-tune YOLO11s at 1280 px and evaluate on held-out *videos*, then promote the model.
   - Consider a temporal heatmap model (TrackNet-style, using three stacked frames) for blurred, tiny balls.
2. **Tracking evaluation on real labelled sequences.** Measure continuity, fragmentation, false associations and position error with the metrics already used for detection.
3. **Automatic pitch and stump localisation.**
   - Train stumps and crease detection, using the Kaggle segmentation set or Roboflow stumps sets.
   - Seed calibration from the detected keypoints, so manual clicks become corrections rather than the only input.
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
