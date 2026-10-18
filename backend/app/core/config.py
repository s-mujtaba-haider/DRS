"""Application settings (environment variables prefixed DRS_, or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DRS_", env_file=ROOT / ".env", extra="ignore")

    storage_dir: Path = ROOT / "storage"
    annotations_dir: Path = ROOT / "data" / "annotations"  # where labelled sets are exported
    database_url: str | None = None  # default: sqlite file in storage_dir
    max_upload_mb: int = 1024
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")
    cors_origins: tuple[str, ...] = ("http://localhost:3100", "http://127.0.0.1:3100")
    worker_threads: int = 1  # one GPU job at a time keeps VRAM bounded
    default_detector: str = "auto"
    default_device: str = "auto"
    max_frames_per_job: int = 3000
    log_level: str = "INFO"
    # Match-day roles. When either PIN is set, every state-changing request needs a valid PIN
    # (header X-DRS-PIN); recording or withdrawing a review decision needs the umpire PIN.
    operator_pin: str | None = None
    umpire_pin: str | None = None
    capture_retention_s: float = 300.0
    capture_max_disk_mb: int = 20000
    capture_encoder: str = "auto"  # auto = h264_nvenc if it opens, else libx264, else mpeg4
    capture_process_per_camera: bool = False  # True scales better for 2+ high-rate cameras (see EVALUATION.md)

    @property
    def videos_dir(self) -> Path:
        return self.storage_dir / "videos"

    @property
    def jobs_dir(self) -> Path:
        return self.storage_dir / "jobs"

    @property
    def capture_dir(self) -> Path:
        return self.storage_dir / "capture"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.storage_dir / 'drs.sqlite3').as_posix()}"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.storage_dir, s.videos_dir, s.jobs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
