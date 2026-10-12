# Datasets

The machine-readable source of truth is [`data/dataset_registry.json`](../data/dataset_registry.json). `scripts/download_datasets.py` updates it with the status, local path and errors. The research was done on 2026-10-12.

## Sources searched

- **Kaggle:** API metadata and listings, verified anonymously.
- **Hugging Face Datasets:** API, gating status checked.
- **GitHub:** API, repository licence checked.
- **Roboflow Universe:** every page returned HTTP 403 to automated fetches, so those entries are **unverified** and come from search snippets only.
- **arXiv / Papers With Code.**
- **IEEE DataPort.**
- **Commercial resellers.**

## Downloaded and used

| ID | What | Licence | Use in this project | Status |
|---|---|---|---|---|
| `hf-cricket-synth` | 410 Blender-simulated deliveries (drag, Magnus, seam, friction bounce) with exact 3D positions, pixel projections and camera intrinsics/extrinsics; about 110 rendered frames from 3 clips | MIT | Trajectory, monocular reconstruction and LBW-projection benchmark (`scripts/evaluate_trajectory_synth.py`); detector-training smoke test | Downloaded (18 MB) |
| `hf-cricket-delivery-videos` | About 260 single-delivery broadcast clips (1920×1080, 25 fps) with JSONL metadata (no ball labels) | Apache-2.0 stated; footage is broadcast-derived, so **research only** | Qualitative real-footage tests of detection and tracking | Metadata plus 15 clips downloaded (43 MB) |
| `kaggle-kushagra-ball-yolo` | 1,912 files: ball images, mostly close-ups (median ball size 66 px; only 5% under 16 px), YOLO labels, class `cricketBall` | CC0-1.0 (image provenance not stated) | **Training the cricket-ball detector.** The original splits **leak**: 112 identical files appear in both train and val/test. The set was rebuilt with `prepare_datasets.py yolo-dir ... --resplit`: 1,800 unique images, grouped by filename and dHash, split 1,440 / 181 / 179 with no cross-split duplicates | Downloaded (376 MB) |
| `kaggle-semantic-seg` | 298 broadcast frames (1920×1080), SuperAnnotate instance masks | CC BY 4.0 (broadcast source) | Converted to boxes with `prepare_datasets.py broadcast-seg`: 213 ball, 284 stumps and 1,230 player boxes. **Used only as an independent broadcast-domain test set**, never for training | Downloaded (551 MB) |
| `kaggle-lbw-wide-noball-videos` | Short broadcast-like clips in Legal / Wide / LBW / NoBall folders | CC0 claimed | Qualitative tests only. Folder labels are not ball-tracking ground truth | Downloaded (159 MB) |
| `hf-jeffrey632-delivery-analysis` | 5 clips with Hawk-Eye-style JSON (release, bounce, impact, stump projection) | Not stated | Format reference only; provenance of the numbers is unknown | Downloaded (57 MB) |

## Verified but blocked or not downloaded

| ID | Why not | How to get it |
|---|---|---|
| `kaggle-wide-balls-2view` (front and side views plus calibration material, Apache 2.0, 978 MB) | Size; not needed until the multi-camera work. The only multi-view set with calibration material; sync is unknown | `--only kaggle-wide-balls-2view` |
| `kaggle-deepsportradar-bowlrelease` (CC BY-NC-ND, 3 GB) | Size, and non-commercial / no-derivatives terms | Manual, if needed for delivery segmentation |
| `hf-cricket-shot`, `hf-suksham-frames`, `hf-infactory-soccer-ball` | Not needed yet, or licence unclear / non-commercial | See registry |
| `github-fpv-balltracking` (3,000 broadcast ball boxes) | Requires downloading YouTube broadcasts, which platform terms and copyright don't permit; no licence file | Not used |
| `github-pradeep-edgefleet` (ball CSVs and weights) | No licence, so not reused | Not used |
| Roboflow projects (appmaker about 9k ball images; lbwdrs ball/bat/pads/wicket; stumps sets) | Unverified (HTTP 403); a free API key is needed; licences must be checked per project | Get a key, check the licence, export YOLO, then `prepare_datasets.py yolo-dir` |

Code repositories checked for reference only (no code copied): `uditarora/cricket-umpire-assistance` (GPL-3.0), `kafle1/drs` (MIT), `opengoalapp/cricket_tracking` (MIT).

## Gaps

No public dataset was found with any of the following:
- Real multi-view, synchronised, high-frame-rate cricket footage with 3D ground truth. Hawk-Eye data is proprietary.
- LBW outcomes paired with ball-tracking ground truth.
- Bat-ball contact ground truth (for example, audio plus high-speed video).

So:
- Real-world LBW accuracy **cannot be measured** with available data.
- Trajectory and LBW layers are evaluated on synthetic physics (cricket-synth).
- Detection on real footage needs either the Kaggle/Roboflow data or human-labelled frames from your own footage. The annotation workflow for that is `scripts/annotate_video.py`, plus UI corrections.

## Reproducing

```powershell
python scripts\download_datasets.py --max-videos 15
python scripts\prepare_datasets.py synth
python scripts\validate_datasets.py --viz 12
```

Prepared datasets record per-sample provenance (`provenance.csv`). The `label_origin` column takes one of these values:
- `human`
- `auto`: unreviewed model output, excluded from training by default.
- `auto_corrected`
- `synthetic_exact`
- `unlabeled`

Splits are made per source sequence (clip or video), never per frame, and validation reports any sequence that leaks across splits.
