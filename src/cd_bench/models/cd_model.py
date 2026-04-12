"""Wrapper encoder + decoder pour l'inférence sur images brutes.

L'encoder est toujours gelé. Le pipeline d'entraînement utilise les
features pré-extraites (memmap) et n'instancie que le décodeur.
Ce wrapper sert uniquement à ``show`` et ``eval --run-id``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cd_bench.models.encoders.base import FeatureEncoder


class CDModel(nn.Module):
    def __init__(self, encoder: FeatureEncoder, decoder: nn.Module):
        super().__init__()
        if encoder.model is None:
            raise RuntimeError("encoder must be loaded (call encoder.load(img_size)) before wrapping")
        self.encoder = encoder
        self.encoder_module = encoder.model
        self.decoder = decoder
        for p in self.encoder_module.parameters():
            p.requires_grad_(False)
        self.encoder_module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder_module.eval()
        return self

    def forward(self, pv1: torch.Tensor, pv2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f1 = self.encoder.forward(pv1)
            f2 = self.encoder.forward(pv2)
        return self.decoder(f1, f2)
