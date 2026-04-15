from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = PACKAGE_DIR.parent
DATA_DIR = SERVICE_ROOT / "data"
MODELS_DIR = SERVICE_ROOT / "models"
SAVED_MODELS_DIR = SERVICE_ROOT / "saved_models"
