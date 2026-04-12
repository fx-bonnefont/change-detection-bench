"""Briques de fusion spatio-temporelle partagées entre décodeurs.

Toute brique qui combine les features ``t1`` et ``t2`` en une seule carte
spatiale vit ici, pour éviter les imports croisés entre décodeurs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RichFusion(nn.Module):
    """Fusion riche spatio-temporelle des patches t1/t2.

    Concatène ``[p1, p2, |p1 - p2|]`` puis projette en ``D_model``.
    Le token CLS est volontairement ignoré : on veut une carte purement
    spatiale pour le décodeur.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(4 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        p1 = t1[:, 1:, :]
        p2 = t2[:, 1:, :]
        diff = torch.abs(p1 - p2)
        prod = p1 * p2
        fusion = torch.cat([p1, p2, diff, prod], dim=-1)
        fused = self.proj(fusion)
        fused = self.act(fused)
        fused = self.norm(fused)
        return fused
