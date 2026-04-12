"""Remapping des classes sémantiques HI-UCD pour le SCD.

Classes brutes 0-9 → classes compactes 0-N_CLASSES.
Les classes à <1% de représentation (4, 6, 7) sont fusionnées dans
``IGNORE_INDEX=0`` (unlabeled).

La LUT numpy permet un remapping vectorisé en une seule indexation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Remapping : raw_class -> compact_class
# 0 -> 0 (unlabeled)
# 1 -> 1, 2 -> 2, 3 -> 3, 5 -> 4, 8 -> 5, 9 -> 6
# 4, 6, 7 -> 0 (supprimés)
_RAW_TO_COMPACT = np.array([0, 1, 2, 3, 0, 4, 0, 0, 5, 6], dtype=np.int64)

N_CLASSES = 6  # classes sémantiques utiles (1..6), 0 = ignore
IGNORE_INDEX = 0


def remap_mask(raw: np.ndarray) -> np.ndarray:
    """Applique la LUT sur un array de classes brutes (0-9) -> (0-N_CLASSES)."""
    return _RAW_TO_COMPACT[raw]


def load_scd_targets(mask_path: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Charge un masque HI-UCD et renvoie ``(sem_t1, sem_t2, valid)`` en ``(H, W)``.

    - ``sem_t1`` : classes sémantiques remappées à T1 (int64, 0..N_CLASSES)
    - ``sem_t2`` : classes sémantiques remappées à T2 (int64, 0..N_CLASSES)
    - ``valid``  : float32, 1 là où canal2 != 0 (pixel annoté)
    """
    with Image.open(mask_path) as im:
        arr = np.array(im, dtype=np.uint8)

    sem_t1 = torch.from_numpy(remap_mask(arr[..., 0]).copy()).long()
    sem_t2 = torch.from_numpy(remap_mask(arr[..., 1]).copy()).long()
    valid = torch.from_numpy((arr[..., 2] != 0).astype(np.float32))

    return sem_t1, sem_t2, valid
