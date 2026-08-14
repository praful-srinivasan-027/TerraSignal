import os
from pathlib import Path
from typing import Set
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or defaults."""

    model_config = ConfigDict(env_file=".env", extra="ignore")

    # Project paths
    PROJECT_ROOT: Path = BASE_DIR
    MODEL_PATH: Path = BASE_DIR / "ML models" / "weights" / "best.pt"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = BASE_DIR / "storage" / "uploads"

    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'inference.db'}"

    # Authentication & Security
    SECRET_KEY: str = "super-secret-jwt-key-change-in-production-123456789"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # File upload validation settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    ALLOWED_MIME_TYPES: Set[str] = {"image/jpeg", "image/png", "image/webp"}

    # ML Inference parameters
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    MODEL_VERSION: str = "yolo_plant_leaf_v1"

    # Concurrency & Worker lifecycle settings
    MAX_RETRIES: int = 3
    WORKER_POLL_INTERVAL: float = 1.0  # seconds
    STALE_JOB_TIMEOUT_SECONDS: int = 600  # 10 minutes

    # Retention & cleanup settings
    RETENTION_HOURS: int = 24


settings = Settings()

# Ensure required storage directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
