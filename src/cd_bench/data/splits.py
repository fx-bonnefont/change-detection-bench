"""Découpage train / val / test partagé par toute l'équipe.

Le ``.dat`` produit par ``cdbench extract`` contient **tous** les tiles
HI-UCD (officiels train + valid) dans un pool unique. Le découpage en
train / val / test est fait *au runtime* à partir de ``metadata["items"]``,
de manière déterministe et stratifiée par densité de pixels ``changed``.

!!!!!! ``SPLIT_SEED`` et les paramètres par défaut sont des **constantes
partagées** : ne PAS les modifier sans accord du groupe, sinon les runs
ne sont plus comparables entre membres.
"""
from __future__ import annotations
import random
from typing import Iterable

# --- Constantes partagées ---------------------------------------------------
SPLIT_SEED: int = 42
SPLIT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)  # train, val, test
N_BINS: int = 5
# ---------------------------------------------------------------------------


def _quantile_bin(values: list[float], n_bins: int) -> list[int]:
    """Affecte chaque valeur à un bin quantile (équi-effectif)."""
    if n_bins <= 1:
        return [0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    bins = [0] * len(values)
    n = len(values)
    for rank, idx in enumerate(order):
        b = min(rank * n_bins // n, n_bins - 1)
        bins[idx] = b
    return bins


def stratified_split(
    items: list[dict],
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
    seed: int = SPLIT_SEED,
    n_bins: int = N_BINS,
) -> dict[str, list[int]]:
    """Split stratifié par ``changed_ratio``.

    Bin les tiles en ``n_bins`` quantiles selon leur ``changed_ratio``,
    puis distribue chaque bin selon ``fractions`` avec une seed fixe.
    Garantit que train/val/test ont une distribution similaire de
    proportions changed/unchanged.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    ratios = [float(it["changed_ratio"]) for it in items]
    bins = _quantile_bin(ratios, n_bins)

    by_bin: dict[int, list[int]] = {}
    for idx, b in enumerate(bins):
        by_bin.setdefault(b, []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    f_train, f_val, _ = fractions
    for b in sorted(by_bin):
        bucket = by_bin[b][:]
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(round(n * f_train))
        n_val = int(round(n * f_val))
        train_idx.extend(bucket[:n_train])
        val_idx.extend(bucket[n_train : n_train + n_val])
        test_idx.extend(bucket[n_train + n_val :])

    return {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }


def split_summary(items: list[dict], splits: dict[str, list[int]]) -> dict[str, dict]:
    """Petit résumé loggable (taille + changed_ratio moyen par split)."""
    out: dict[str, dict] = {}
    for name, idxs in splits.items():
        if not idxs:
            out[name] = {"n": 0, "mean_changed_ratio": 0.0}
            continue
        rs = [float(items[i]["changed_ratio"]) for i in idxs]
        out[name] = {
            "n": len(idxs),
            "mean_changed_ratio": sum(rs) / len(rs),
        }
    return out
