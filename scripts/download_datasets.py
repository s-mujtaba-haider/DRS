"""Download datasets listed in data/dataset_registry.json (entries with auto_download=true).

* Hugging Face datasets are fetched anonymously with huggingface_hub.
* Kaggle datasets need a Kaggle API token (KAGGLE_USERNAME/KAGGLE_KEY env vars or
  %USERPROFILE%/.kaggle/kaggle.json). Without it they are skipped and recorded as blocked.

Failures are recorded in the registry ("status", "last_error") and the script moves on.

Usage:
    python scripts/download_datasets.py                 # all auto-download entries
    python scripts/download_datasets.py --only hf-cricket-synth
    python scripts/download_datasets.py --max-videos 20 # subset size for large video sets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "dataset_registry.json"
RAW = ROOT / "data" / "raw"


def kaggle_credentials_present() -> bool:
    # New-style access tokens (KGAT_...) via KAGGLE_API_TOKEN or ~/.kaggle/access_token,
    # or legacy username/key via env vars or ~/.kaggle/kaggle.json.
    if os.environ.get("KAGGLE_API_TOKEN") or (Path.home() / ".kaggle" / "access_token").is_file():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


def hf_repo_id(url: str) -> str:
    return url.split("huggingface.co/datasets/")[1].strip("/")


def download_hf(entry: dict, dest: Path, max_videos: int) -> str:
    from huggingface_hub import HfApi, snapshot_download

    repo = hf_repo_id(entry["url"])
    if entry["id"] == "hf-cricket-delivery-videos":
        # ~1 GB set: fetch metadata plus a deterministic subset of test-split clips.
        files = HfApi().list_repo_files(repo, repo_type="dataset")
        meta = [f for f in files if f.endswith((".jsonl", ".md", ".json"))]
        videos = sorted(f for f in files if f.lower().endswith(".mp4"))
        test_vids = [f for f in videos if "test" in f.lower()] or videos
        chosen = test_vids[:max_videos]
        snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=meta + chosen)
        return f"metadata + {len(chosen)} of {len(videos)} clips"
    snapshot_download(repo, repo_type="dataset", local_dir=dest)
    return "full snapshot"


def download_kaggle(entry: dict, dest: Path) -> str:
    if not kaggle_credentials_present():
        raise PermissionError(
            "Kaggle API token not configured. Create one at kaggle.com > Settings > API, then either set "
            "KAGGLE_API_TOKEN, save it to %USERPROFILE%\\.kaggle\\access_token, or (legacy) save kaggle.json to "
            "%USERPROFILE%\\.kaggle\\kaggle.json, and re-run this script."
        )
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    ref = entry["download_method"].split("-d ")[1].strip()
    api.dataset_download_files(ref, path=str(dest), unzip=False, quiet=False)
    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():  # zip-slip guard
                if Path(member).is_absolute() or ".." in Path(member).parts:
                    raise ValueError(f"Unsafe path in archive: {member}")
            zf.extractall(dest)
        z.unlink()
    return "full archive"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="registry ids to download")
    ap.add_argument("--max-videos", type=int, default=15)
    args = ap.parse_args()

    reg = json.loads(REGISTRY.read_text())
    for entry in reg["datasets"]:
        if args.only and entry["id"] not in args.only:
            continue
        if not args.only and not entry.get("auto_download"):
            continue
        dest = RAW / entry["id"]
        dest.mkdir(parents=True, exist_ok=True)
        print(f"==> {entry['id']} ({entry['source']})")
        try:
            if entry["source"] == "huggingface":
                note = download_hf(entry, dest, args.max_videos)
            elif entry["source"] == "kaggle":
                note = download_kaggle(entry, dest)
            else:
                raise NotImplementedError(f"No automatic downloader for source '{entry['source']}'")
            n_files = sum(1 for p in dest.rglob("*") if p.is_file())
            size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
            entry.update(status="downloaded", local_path=str(dest.relative_to(ROOT)).replace("\\", "/"),
                         download_note=f"{note}; {n_files} files, {size / 1e6:.1f} MB",
                         downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"), last_error=None)
            print(f"    ok: {entry['download_note']}")
        except PermissionError as exc:
            entry.update(status="blocked_credentials", last_error=str(exc))
            print(f"    blocked: {exc}")
        except Exception as exc:  # record and continue with other datasets
            entry.update(status="download_failed", last_error=f"{type(exc).__name__}: {exc}")
            print(f"    FAILED: {exc}")
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
