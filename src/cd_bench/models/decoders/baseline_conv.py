"""Tête convolutionnelle simple (baseline) pour le SCD.

Sert de point de comparaison "léger" face au ``ChangeQueryDecoder`` :
même fusion riche en entrée, même cible (carte 512×512), mais pas de
mécanisme d'attention — uniquement quelques convolutions 2D et un
upsampling apprenable pour passer de la grille de patches à la
résolution pleine. Sort ``2*K`` canaux (K par date).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cd_bench.data.mask_mapping import N_CLASSES
from cd_bench.models.decoders.fusion import RichFusion

OUT_CHANNELS = 2 * (N_CLASSES + 1)  # K classes pour T1 + K pour T2


class _UpBlock(nn.Module):
    """Upsampling apprenable ×2 via ConvTranspose2d + GELU.

    Remplace l'ancienne combinaison ``F.interpolate + Conv2d 3×3`` qui
    déclenchait un chemin backward catastrophiquement lent sur MPS
    (~12× le forward). ``ConvTranspose2d`` est une op fusionnée bien
    optimisée par Metal, et apprend toujours un upsampling 2× — pattern
    standard des U-Net et décodeurs modernes.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.up(x))


class BaselineConvHead(nn.Module):
    """Décodeur conv-only avec upsampling apprenable."""

    trainable = True

    def __init__(self, d_model: int, hidden: int = 128, out_size: int = 512, out_channels: int = OUT_CHANNELS):
        super().__init__()
        self.d_model = d_model
        self.out_size = out_size
        self.out_channels = out_channels

        self.fusion = RichFusion(d_model)

        # Projection initiale à la résolution patch (side × side).
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Upsampling apprenable : 2 ×2 = ×4, qui amène patch-16 (side=32)
        # à 128×128. Le passage final 128 -> out_size est fait par un
        # ``F.interpolate(nearest)``.
        c1, c2 = hidden, max(hidden // 2, 8)
        self.up = nn.Sequential(
            _UpBlock(hidden, c1),
            _UpBlock(c1, c2),
        )
        self.head = nn.Conv2d(c2, out_channels, kernel_size=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        feats = self.fusion(t1, t2)
        b, n, d = feats.shape
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError(f"N_patches={n} n'est pas un carré parfait")

        x = feats.transpose(1, 2).reshape(b, d, side, side)
        x = self.proj(x)
        x = self.up(x)
        logits = self.head(x)

        # Upscaling final non-apprenable 128 -> 512 (×4 nearest).
        if logits.shape[-1] != self.out_size:
            logits = F.interpolate(
                logits, size=(self.out_size, self.out_size), mode="nearest"
            )
        return logits
