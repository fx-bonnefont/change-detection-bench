"""Encoder convolutionnel / hiérarchique : sortie spatiale ``(B, C, H, W)``.

Couvre ConvNeXt, ResNet et plus généralement tout modèle HF dont la
sortie principale est une carte de features 2D plutôt qu'une séquence
de tokens. Pour respecter le contrat du pipeline (CLS + patches), on :

    1. flatten la grille en ``(B, H*W, C)``
    2. synthétise un CLS = moyenne globale des patches, prepend à l'index 0

⚠️ La grille spatiale doit être **carrée** ; les modèles purement
hiérarchiques type Swin (downsampling asymétrique sur résolution non
divisible) feront échouer le ``_self_test`` au load.
"""
from __future__ import annotations

import torch

from cd_bench.models.encoders.base import FeatureEncoder, get_hidden_size


class ConvLikeEncoder(FeatureEncoder):
    def _infer_shape(self, img_size: int) -> tuple[int, int]:
        dim = get_hidden_size(self.model.config)
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, img_size, img_size, device=self.device)
            feat = self._extract_feature_map(dummy)
        h, w = feat.shape[-2], feat.shape[-1]
        return dim, 1 + h * w

    def _extract_feature_map(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Récupère la carte ``(B, C, H, W)`` de la dernière étape."""
        out = self.model(pixel_values)
        if hasattr(out, "feature_maps") and out.feature_maps is not None:
            return out.feature_maps[-1]
        if hasattr(out, "last_hidden_state") and out.last_hidden_state.ndim == 4:
            return out.last_hidden_state
        raise RuntimeError(
            f"{type(self).__name__}: impossible de localiser une carte "
            f"spatiale (B,C,H,W) dans la sortie du modèle."
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self._extract_feature_map(pixel_values)         # (B, C, H, W)
        b, c, h, w = feat.shape
        patches = feat.flatten(2).transpose(1, 2)              # (B, H*W, C)
        cls = patches.mean(dim=1, keepdim=True)                # (B, 1, C)
        return torch.cat([cls, patches], dim=1)                # (B, 1+H*W, C)
