"""Configuration centralisée lue depuis ``configs/settings.toml``.

Après ``git clone`` + ``uv sync`` :
    cp configs/settings.toml.example configs/settings.toml
    # éditer les chemins
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import]
    except ImportError:
        import tomli as tomllib  # type: ignore[import,no-redef]

SETTINGS_PATH = Path(__file__).resolve().parents[2] / "configs" / "settings.toml"

if not SETTINGS_PATH.exists():
    raise FileNotFoundError(
        f"Fichier de configuration introuvable : {SETTINGS_PATH}\n"
        "→ cp configs/settings.toml.example configs/settings.toml  puis édite les chemins."
    )

with open(SETTINGS_PATH, "rb") as f:
    _cfg = tomllib.load(f)

# --- Data paths ---
RAW_DATA_DIR = Path(_cfg["data"]["raw_data_dir"])
EXTRACTED_FEATURES_DIR = Path(_cfg["data"]["features_dir"])

# --- MLflow ---
MLFLOW_TRACKING_URI = _cfg["mlflow"]["tracking_uri"]
MLFLOW_DB_PATH = Path(_cfg["mlflow"]["db_path"])
MLFLOW_ARTIFACTS_DIR = Path(_cfg["mlflow"]["artifacts_dir"])


# --- Helpers encodeur (inchangés) ---
def storage_name(encoder_name: str, limit: int | None = None) -> str:
    return f"{encoder_name}-limit{limit}" if limit else encoder_name


def encoder_dir(encoder_name: str, limit: int | None = None) -> Path:
    return EXTRACTED_FEATURES_DIR / storage_name(encoder_name, limit)


def dat_path(encoder_name: str, limit: int | None = None) -> Path:
    name = storage_name(encoder_name, limit)
    return encoder_dir(encoder_name, limit) / f"data_{name}.dat"


def metadata_path(encoder_name: str, limit: int | None = None) -> Path:
    name = storage_name(encoder_name, limit)
    return encoder_dir(encoder_name, limit) / f"metadata_{name}.json"
