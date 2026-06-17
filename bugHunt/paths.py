"""Shared filesystem paths for local and Kaggle training runs."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


BASE_DIR = env_path("BUGHUNT_BASE_DIR", PROJECT_DIR.parent)
RAW_DIR = env_path("BUGHUNT_RAW_DIR", BASE_DIR / "data" / "raw")
PROCESSED_DIR = env_path("BUGHUNT_PROCESSED_DIR", BASE_DIR / "data" / "processed")
MODELS_DIR = env_path("BUGHUNT_MODELS_DIR", BASE_DIR / "models")
LOGS_DIR = env_path("BUGHUNT_LOGS_DIR", BASE_DIR / "logs")
CHECKPOINTS_DIR = env_path("BUGHUNT_CHECKPOINTS_DIR", LOGS_DIR / "checkpoints")
