"""Visualisation des prédictions SCD."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None  # type: ignore[assignment]

from cd_bench.data.mask_mapping import N_CLASSES

# Palette HI-UCD officielle, indexée par classe compacte (après remap).
# compact 0=unlabeled, 1=Water, 2=Grass, 3=Building, 4=Road, 5=Bare land, 6=Woodland
_COLORS = np.array([
    [255, 255, 255],  # 0: unlabeled
    [  0, 153, 255],  # 1: Water
    [202, 255, 122],  # 2: Grass
    [230,   0,   0],  # 3: Building
    [255, 230,   0],  # 4: Road       (raw 5)
    [175, 122, 255],  # 5: Bare land  (raw 8)
    [ 26, 255,   0],  # 6: Woodland   (raw 9)
], dtype=np.uint8)

CLASS_NAMES = ["unlabeled", "Water", "Grass", "Building", "Road", "Bare land", "Woodland"]


def colorize_semantic(sem_map: np.ndarray) -> np.ndarray:
    """Convertit une carte sémantique (H, W) int en image RGB (H, W, 3)."""
    return _COLORS[sem_map]


def make_figure(
    img_2018: Image.Image,
    img_2019: Image.Image,
    sem_t1: np.ndarray,
    sem_t2: np.ndarray,
    change_mask: np.ndarray,
    tile_id: str,
) -> "plt.Figure":
    """Crée une figure 2×2 : images brutes + cartes sémantiques prédites.

    Layout : [2018 brut | 2019 brut]
             [sem T1    | sem T2 + contour changement]
    """
    if plt is None:
        raise ImportError("matplotlib est requis pour la visualisation.")

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].imshow(np.array(img_2018.convert("RGB")))
    axes[0, 0].set_title("2018")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.array(img_2019.convert("RGB")))
    axes[0, 1].set_title("2019")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(colorize_semantic(sem_t1))
    axes[1, 0].set_title("Semantic T1 (predicted)")
    axes[1, 0].axis("off")

    sem_t2_rgb = colorize_semantic(sem_t2)
    axes[1, 1].imshow(sem_t2_rgb)
    if change_mask.any():
        axes[1, 1].contour(change_mask.astype(float), levels=[0.5], colors="white", linewidths=0.8)
    axes[1, 1].set_title("Semantic T2 + change contour")
    axes[1, 1].axis("off")

    # Légende des classes
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=np.array(_COLORS[i]) / 255.0, label=CLASS_NAMES[i])
        for i in range(1, len(CLASS_NAMES))  # skip unlabeled
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches), fontsize=9)

    pct_changed = change_mask.mean() * 100
    fig.suptitle(f"{tile_id} — predicted change: {pct_changed:.1f}%", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig


def save_figure(fig: "plt.Figure", path: Path) -> None:
    """Sauvegarde la figure en PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
