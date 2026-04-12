"""Query-based change-detection decoder.

Décodeur inspiré de Mask2Former / DETR : un petit jeu de requêtes
apprenables va piocher, via attention croisée, dans une carte de features
spatio-temporelle obtenue par fusion riche (concat + diff absolue) des
features encodeur de t1 et t2.

Les features d'entrée ont la forme ``(B, N_tokens, D_model)`` où le token
0 est le CLS et les ``N_tokens - 1`` suivants sont les patches d'une grille
carrée ``H = W = sqrt(N_patches)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cd_bench.data.mask_mapping import N_CLASSES
from cd_bench.models.decoders.fusion import RichFusion

OUT_CHANNELS = 2 * (N_CLASSES + 1)


class PositionalEncoding2D(nn.Module):
    """Encodage positionnel 2D sinus/cosinus pour une grille carrée de patches."""

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model doit être divisible par 4, reçu {d_model}")
        self.d_model = d_model
        self._cache: dict[int, torch.Tensor] = {}

    def _build(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        d_half = self.d_model // 2

        div_term = torch.exp(
            torch.arange(0, d_half, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_half)
        )

        y_pos = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
        x_pos = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1)

        pe_y = torch.zeros(h, d_half, device=device, dtype=torch.float32)
        pe_x = torch.zeros(w, d_half, device=device, dtype=torch.float32)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)

        pe = torch.zeros(h, w, self.d_model, device=device, dtype=torch.float32)
        pe[:, :, :d_half] = pe_y.unsqueeze(1).expand(h, w, d_half)
        pe[:, :, d_half:] = pe_x.unsqueeze(0).expand(h, w, d_half)
        return pe.reshape(h * w, self.d_model).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError(f"N_patches={n} n'est pas un carré parfait")
        key = side
        pe = self._cache.get(key)
        if pe is None or pe.device != x.device or pe.dtype != x.dtype:
            pe = self._build(side, side, x.device, x.dtype)
            self._cache[key] = pe
        return x + pe.unsqueeze(0)


class ChangeQueryDecoder(nn.Module):
    """Décodeur Mask2Former-like pour la change detection."""

    trainable = True

    def __init__(
        self,
        d_model: int,
        num_queries: int = 128,
        num_layers: int = 3,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        out_size: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.out_size = out_size

        self.fusion = RichFusion(d_model)
        self.pos_enc = PositionalEncoding2D(d_model)

        self.queries = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head = nn.Conv2d(num_queries, OUT_CHANNELS, kernel_size=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        feats = self.fusion(t1, t2)
        feats = self.pos_enc(feats)

        b, n, d = feats.shape
        side = int(round(math.sqrt(n)))

        q = self.queries.weight.unsqueeze(0).expand(b, -1, -1)
        q_out = self.decoder(tgt=q, memory=feats)

        sim = torch.einsum("bqd,bnd->bqn", q_out, feats) / math.sqrt(d)
        sim = sim.reshape(b, self.num_queries, side, side)

        logits = self.head(sim)
        logits = F.interpolate(
            logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )
        return logits
