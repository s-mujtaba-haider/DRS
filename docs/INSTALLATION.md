# Installation (Windows)

## Requirements

- Windows 10/11, Python 3.12 (3.11+), Node.js 20+ (developed on 24), Git.
- Optional NVIDIA GPU with a recent driver. Tested on an RTX 3080 Ti, driver 616.56, with the CUDA 12.8 PyTorch wheels. No system CUDA Toolkit is needed: the wheels bundle CUDA.
- Optional Docker Desktop with the WSL2 backend, for the containerised GPU backend.
- FFmpeg CLI is **not** required: PyAV bundles the FFmpeg libraries, including the libx264 encoder.

## 1. Python environment

```powershell
cd DRS
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements\backend.txt -r requirements\ml-gpu.txt -r requirements\dev.txt
.\.venv\Scripts\python scripts\inspect_environment.py
```

For a CPU-only setup, skip the torch line and `ml-gpu.txt`. The app still runs: detection falls back to the motion heuristic, and every other module is NumPy/SciPy/OpenCV.

### Smart App Control (`WinError 4551 ... c10.dll`)

On Windows 11 with Smart App Control enforced, the first import of PyTorch can fail because its DLLs are unsigned. During development the block cleared later, once reputation had been established. If it persists:
- Use the Docker backend (section 4). It is not affected.
- Or decide for yourself whether to change Smart App Control. On many builds it **can't be turned back on** without reinstalling Windows, and this project does not require it.

## 2. Data and weights

```powershell
.\.venv\Scripts\python scripts\download_datasets.py          # anonymous Hugging Face sets
.\.venv\Scripts\python -c "from ultralytics import YOLO; YOLO('models/checkpoints/yolo11s.pt')"   # generic COCO weights
```

Kaggle datasets need an API token (from Kaggle, Settings, API, Create New Token). Supply it in one of three ways:
- set `KAGGLE_API_TOKEN` in the shell that runs the script;
- save it to `%USERPROFILE%\.kaggle\access_token`;
- for legacy keys, use `kaggle.json`.

Never commit it. Revoke and recreate it if it is ever exposed.

To rebuild the cricket-ball detector, run the commands in `docs/MODEL_CARD.md`. It takes about 30 minutes on an RTX 3080 Ti.

## 3. Run

Backend (port 8100, configurable):

```powershell
copy .env.example .env      # optional
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8100
```

Health and docs:
- http://127.0.0.1:8100/api/health
- http://127.0.0.1:8100/api/system/status (GPU, torch, models)
- http://127.0.0.1:8100/docs (OpenAPI)

Frontend (port 3100, chosen to avoid the common default 3000):

```powershell
cd frontend
npm install
echo NEXT_PUBLIC_API_BASE=http://127.0.0.1:8100 > .env.local   # only if the backend is elsewhere
npm run build
npm run start      # or: npm run dev
```

Open http://localhost:3100.

To change ports:
- Backend: use `--port` on uvicorn and set `DRS_CORS_ORIGINS` to match.
- Frontend: use `npx next start -p <port>` and update `NEXT_PUBLIC_API_BASE` and `DRS_CORS_ORIGINS` accordingly.

## 4. Docker GPU backend (optional)

```powershell
docker compose up --build        # backend on :8100 with --gpus; frontend still runs on the host
```

The image is based on `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`.
- Volumes: `data/`, `models/` and `outputs/` are mounted; storage goes to `storage-docker/`.
- Check GPU access with `docker run --rm --gpus all drs-backend-gpu:local python -c "import torch;print(torch.cuda.is_available())"`.

## 5. WSL2 (optional)

The code is cross-platform. Inside a WSL2 Ubuntu you can follow the same steps with `python3 -m venv` and the CUDA 12.8 wheels. A WSL distribution is not required.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI says "Backend unreachable" | Start uvicorn, check `NEXT_PUBLIC_API_BASE`, and make sure the port isn't used (`netstat -ano \| findstr 8100`) |
| CORS error in the browser console | Add your frontend origin to `DRS_CORS_ORIGINS` |
| Job fails with "Cricket ball weights not found" | Expected until you train a model. Choose `auto`, `coco_hybrid` or `motion` |
| CUDA out of memory | Reduce `batch_size`. The detector halves the batch automatically and falls back to CPU at batch 1 |
| Upload rejected with 413 / 415 / 422 | The file is too large (`DRS_MAX_UPLOAD_MB`), has an unsupported extension, or isn't decodable |
| Video plays but overlays drift | Variable-frame-rate source: overlays use per-frame PTS, so check the VFR warning in the metadata |
| `npm install` fails with EALLOWSCRIPTS | npm 11 install-script policy. Approved scripts are listed in `frontend/package.json` (`allowScripts`) |
