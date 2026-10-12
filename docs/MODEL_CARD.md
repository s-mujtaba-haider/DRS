# Model card

## Models actually available

| Model | Source | Trained on | Status |
|---|---|---|---|
| `yolo11s.pt` (COCO) | Ultralytics release, **AGPL-3.0** | COCO 2017. Uses class 32, "sports ball". | Used as a *generic fallback*. **Not cricket-trained.** On real broadcast clips it detects the ball in some close-ups (up to 0.90 confidence, observed), but it also fires on TV logos and misses the tiny ball in wide shots. |
| Motion candidate heuristic `motion:3frame-diff-v1` | This repository | Not learned | A candidate generator only. Produces many false positives on real footage. |
| `models/runs/smoke_synth` | `train_detector.py --smoke` | 46 synthetic frames, 2 epochs | **Pipeline smoke test only.** Test-split precision and recall are 0 (`outputs/reports/detector_eval_synth_smoke_test.json`). Never promoted. |
| `cricket_ball_best.pt` | — | — | **Does not exist yet.** Requires the training data described in the README. |

The app chooses the detector at runtime and reports the choice in `/api/system/status`, in every job summary and in the UI. When no cricket-trained detector is installed, the UI says so.

## Intended use of a trained detector

- **Task:** detect a cricket ball (single class) in video frames for downstream tracking.
- **Recommended starting configuration:** fine-tune YOLO11s at 1280 px with batch 8 and AMP. Training uses about 3–5 GB of VRAM on the RTX 3080 Ti.
- **Tiling:** optional tiled inference (`tile_size`) keeps tiny balls at native resolution.

## Required before trusting any detector

- Evaluate on a **held-out split by source video** using `evaluate_detector.py`. It reports precision, recall, F1, AP (centre match), localisation error and small-object fraction.
- Also evaluate on labelled frames from **your own target footage**. A different camera or broadcaster is a different distribution.
- Record the run: `train_detector.py` writes `run_record.json` (config and library versions) and `model_card.json` (validation metrics).

## Licensing

Fine-tuning Ultralytics weights inherits AGPL-3.0 obligations, which apply if you distribute the model or offer it as a network service. A non-AGPL path is to train a different detector, such as RT-DETR from other sources or a custom heatmap network, behind the same `BallDetector` interface.
