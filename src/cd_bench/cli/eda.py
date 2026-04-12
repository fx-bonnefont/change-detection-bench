"""Sous-commande ``cdbench eda`` : analyse exploratoire des masques HI-UCD.

Scanne les masques PNG (3 canaux) et affiche un tableau récapitulatif :
valeurs uniques et ratios par canal, ventilés par période (T1/T2) pour
les canaux sémantiques.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import typer
from PIL import Image

from cd_bench.config import RAW_DATA_DIR
from cd_bench.data.paths import get_paths


def _scan_masks(mask_paths: list, limit: int | None = None) -> dict[int, Counter]:
    """Retourne {canal: Counter(valeur -> nb_pixels)} sur tous les masques."""
    counters: dict[int, Counter] = {0: Counter(), 1: Counter(), 2: Counter()}
    paths = mask_paths[:limit] if limit else mask_paths
    for p in paths:
        arr = np.array(Image.open(p))  # (H, W, 3)
        for ch in range(3):
            vals, counts = np.unique(arr[:, :, ch], return_counts=True)
            for v, c in zip(vals, counts):
                counters[ch][int(v)] += int(c)
    return counters


def _format_table(
    counters_train: dict[int, Counter],
    counters_valid: dict[int, Counter],
) -> str:
    """Construit un tableau texte avec valeurs uniques et ratios par canal."""
    channel_names = {
        0: "Canal 0 (semantic T1)",
        1: "Canal 1 (semantic T2)",
        2: "Canal 2 (change BCD)",
    }

    lines: list[str] = []
    for ch in range(3):
        ct = counters_train[ch]
        cv = counters_valid[ch]
        all_vals = sorted(set(ct.keys()) | set(cv.keys()))
        total_t = sum(ct.values()) or 1
        total_v = sum(cv.values()) or 1

        lines.append("")
        lines.append(f"  {channel_names[ch]}")
        lines.append(f"  {'':>6} {'Valeurs uniques':>18}")
        lines.append(f"  {'-' * 72}")

        # Header
        header = f"  {'Value':>6} | {'Count train':>14} {'% train':>10} | {'Count valid':>14} {'% valid':>10}"
        lines.append(header)
        lines.append(f"  {'-' * 72}")

        for v in all_vals:
            c_t = ct.get(v, 0)
            c_v = cv.get(v, 0)
            pct_t = c_t / total_t * 100
            pct_v = c_v / total_v * 100
            lines.append(
                f"  {v:>6} | {c_t:>14,} {pct_t:>9.2f}% | {c_v:>14,} {pct_v:>9.2f}%"
            )

    return "\n".join(lines)


def eda(
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Nombre max de masques à scanner."),
):
    """Analyse exploratoire des masques HI-UCD (3 canaux)."""
    paths = get_paths(RAW_DATA_DIR, limit=None)

    train_masks = paths["train"]["mask"]
    valid_masks = paths["valid"]["mask"]

    if not train_masks:
        raise typer.BadParameter("Aucun masque trouvé dans le split train.")

    n_train = len(train_masks[:limit]) if limit else len(train_masks)
    n_valid = len(valid_masks[:limit]) if limit else len(valid_masks)
    typer.echo(f"Scanning {n_train} train + {n_valid} valid masks...")

    counters_train = _scan_masks(train_masks, limit)
    counters_valid = _scan_masks(valid_masks, limit)

    typer.echo(_format_table(counters_train, counters_valid))
