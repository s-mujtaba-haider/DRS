# Model card

## Models actually available

| Model | Source | Trained on | Status |
|---|---|---|---|
| `yolo11s.pt` (COCO) | Ultralytics release, **AGPL-3.0** | COCO 2017. Uses class 32, "sports ball". | Used as a *generic fallback*. **Not cricket-trained.** On real broadcast clips it detects the ball in some close-ups (up to 0.90 confidence, observed), but it also fires on TV logos and misses the tiny ball in wide shots. |
| Motion candidate heuristic `motion:3frame-diff-v1` | This repository | Not learned | A candidate generator only. Produces many false positives on real footage. |
| `models/runs/smoke_synth` | `train_detector.py --smoke` | 46 synthetic frames, 2 epochs | **Pipeline smoke test only.** Test-split precision and recall are 0 (`outputs/reports/detector_eval_synth_smoke_test.json`). Never promoted. |
| `cricket_ball_v1_kaggle.pt` (v1, = `models/runs/ball_kaggle_y11s_1280/weights/best.pt`) | YOLO11s fine-tuned in this repo (AGPL-3.0 base) | Kaggle `kushagra3204/cricket-ball-dataset-for-yolo` (CC0): 1,440 train / 181 val / 179 test after the leak-free re-split; 60 epochs at 1280 px, batch 8, AMP; 28.5 min on the RTX 3080 Ti | Superseded by v2. Kept for comparison. Results below. |
| **`cricket_ball_best.pt` = v2** (`models/runs/ball_mix_v2_ft960`) | v1 fine-tuned in this repo | Kaggle + 1,018 track-verified broadcast pseudo-label crops, 960 px, 50 epochs | **Active detector.** Broadcast test at 1280: P 0.62 / R 0.53 / AP 0.49 (v1: 0.64 / 0.37 / 0.38). Details in `docs/EVALUATION.md` §4d |

The app chooses the detector at runtime and reports the choice in `/api/system/status`, in every job summary and in the UI. When no cricket-trained detector is installed, the UI says so.

## v1 (`cricket_ball_v1_kaggle.pt`): measured results

All results use a score threshold of 0.25 and centre-distance matching (≤ max(6 px, half the ball size)). Full numbers are in `outputs/reports/detector_eval_*.json` and `models/checkpoints/cricket_ball_best.json`.

| Test set | Ball size (median) | Precision | Recall | F1 | AP (centre) | Centre error (median) |
|---|---|---|---|---|---|---|
| Kaggle held-out test (same source as training, leak-free) | 69 px | 0.91 | 0.98 | 0.94 | 0.99 | 2.5 px |
| **Independent broadcast frames** (Acme AI masks, IND v ZIM 2022, 298 frames) at 1280 px full frame | 13 px (70% < 16 px) | 0.64 | **0.37** | 0.47 | 0.38 | 1.3 px |
| Same broadcast set, tiled 640 px | 13 px | 0.46 | 0.48 | 0.47 | 0.37 | — |

How to read these numbers:
- In-domain performance is high.
- **On broadcast footage the model misses most balls.** The training images are mostly close-ups, while broadcast wide shots show a 7–13 px ball. Tiling trades precision for recall without improving F1.
- On 5 unlabelled broadcast delivery clips, candidate tracks dropped from 80–130 per clip (generic COCO + motion) to 3–8.
- Two of the five clips produced a primary track passing the consistency checks. One was visually confirmed as the real ball (17 of 18 frames observed). The other three were still flagged unreliable.
- **Next improvement:** add broadcast-domain training data with small balls, labelled from your own footage (`annotate_video.py`) or from licence-checked Roboflow sets. Then re-evaluate on the broadcast test set, which must never be used for training.

Reproduce (needs a Kaggle token in `KAGGLE_API_TOKEN` or `%USERPROFILE%\.kaggle\access_token`):

```powershell
python scripts\download_datasets.py --only kaggle-kushagra-ball-yolo kaggle-semantic-seg
python scripts\prepare_datasets.py yolo-dir data\raw\kaggle-kushagra-ball-yolo --name ball_kaggle --origin human --resplit
python scripts\prepare_datasets.py broadcast-seg
python scripts\train_detector.py --data data\processed\ball_kaggle\data.yaml --epochs 60 --imgsz 1280 --batch 8 --name ball_kaggle_y11s_1280
python scripts\evaluate_detector.py --data data\processed\ball_kaggle\data.yaml --split test --weights models\runs\ball_kaggle_y11s_1280\weights\best.pt --name kaggle_test_y11s_1280
python scripts\evaluate_detector.py --data data\processed\broadcast_seg_test\data.yaml --split test --weights models\runs\ball_kaggle_y11s_1280\weights\best.pt --name broadcast_y11s_1280
copy models\runs\ball_kaggle_y11s_1280\weights\best.pt models\checkpoints\cricket_ball_best.pt
```

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
