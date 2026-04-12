"""Encoder ViT avec register tokens (DINOv2-reg, DINOv3, …).

Layout de ``last_hidden_state`` :

    [CLS] [reg_1 .. reg_N] [patch_1 .. patch_K]

On garde CLS + patches et on **drop** les ``num_registers`` tokens
intercalés, pour respecter le contrat ``(B, 1 + K, D)`` du pipeline.
"""
from __future__ import annotations

import torch

from cd_bench.models.encoders.base import FeatureEncoder, get_hidden_size


class ViTRegLikeEncoder(FeatureEncoder):
    def __init__(self, name: str, hf_id: str, num_registers: int = 4):
        super().__init__(name, hf_id)
        self.num_registers = num_registers

    def _infer_shape(self, img_size: int) -> tuple[int, int]:
        dim = get_hidden_size(self.model.config)
        n_patches = (img_size // self.model.config.patch_size) ** 2
        return dim, 1 + n_patches  # CLS + patches (registers exclus)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        last = self.model(pixel_values).last_hidden_state
        start = 1 + self.num_registers
        return torch.cat([last[:, 0:1, :], last[:, start:, :]], dim=1)
