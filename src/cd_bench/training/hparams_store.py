"""Store partagé des hyperparamètres tunés par ``cdbench tune``.

Format JSON plat (versionné dans git, à ``configs/loss_hparams.json``) :

    {
      "<encoder>__<decoder>": {
        "lambda_bcd": float, "bcd_alpha": float, "bcd_gamma": float,
        "lr": float,
        "best_score": float, "tuned_at": "YYYY-MM-DD",
        "n_trials": int, "epochs_per_trial": int
      },
      ...
    }
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

HPARAMS_STORE_PATH = Path(__file__).resolve().parents[3] / "configs" / "loss_hparams.json"

LOSS_KEYS = ("lambda_bcd", "bcd_alpha", "bcd_gamma")
OPTIM_KEYS = ("lr",)
ALL_HPARAM_KEYS = LOSS_KEYS + OPTIM_KEYS


def make_key(encoder: str, decoder: str) -> str:
    return f"{encoder}__{decoder}"


def load_store(path: Path = HPARAMS_STORE_PATH) -> dict:
    if not path.exists():
        return {}
    text = path.read_text().strip()
    if not text:
        return {}
    return json.loads(text)


def save_store(store: dict, path: Path = HPARAMS_STORE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(store, f, indent=2, sort_keys=True)
        f.write("\n")


def get_loss_kwargs(
    encoder: str, decoder: str, path: Path = HPARAMS_STORE_PATH
) -> dict | None:
    """Retourne ``{lambda_bcd, bcd_alpha, bcd_gamma}`` ou ``None`` si non tuné."""
    store = load_store(path)
    entry = store.get(make_key(encoder, decoder))
    if entry is None:
        return None
    return {k: float(entry[k]) for k in LOSS_KEYS if k in entry}


def get_lr(encoder: str, decoder: str, path: Path = HPARAMS_STORE_PATH) -> float | None:
    """Retourne le LR tuné ou ``None`` si non tuné."""
    store = load_store(path)
    entry = store.get(make_key(encoder, decoder))
    if entry is None or "lr" not in entry:
        return None
    return float(entry["lr"])


def upsert_if_better(
    encoder: str,
    decoder: str,
    loss_kwargs: dict,
    best_score: float,
    n_trials: int,
    epochs_per_trial: int,
    path: Path = HPARAMS_STORE_PATH,
) -> tuple[bool, float | None]:
    """Met à jour l'entrée si ``best_score`` est meilleur que l'existant."""
    store = load_store(path)
    key = make_key(encoder, decoder)
    prev = store.get(key)
    prev_iou = float(prev["best_score"]) if prev else None

    if prev_iou is not None and best_score <= prev_iou:
        logger.info(
            "[hparams_store] kept previous values for %s (%.4f >= %.4f)",
            key, prev_iou, best_score,
        )
        return False, prev_iou

    store[key] = {
        **{k: float(loss_kwargs[k]) for k in ALL_HPARAM_KEYS if k in loss_kwargs},
        "best_score": float(best_score),
        "tuned_at": date.today().isoformat(),
        "n_trials": int(n_trials),
        "epochs_per_trial": int(epochs_per_trial),
    }
    save_store(store, path)
    if prev_iou is None:
        logger.info("[hparams_store] inserted new entry for %s (best_score=%.4f)", key, best_score)
    else:
        logger.info("[hparams_store] updated %s (%.4f -> %.4f)", key, prev_iou, best_score)
    return True, prev_iou
