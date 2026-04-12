"""Encoder hiérarchique à sortie séquentielle : Swin, SwinV2, FocalNet, …

Ces modèles produisent ``last_hidden_state`` de forme ``(B, N, C)`` où
``N = H'·W'`` correspond à la grille downsamplée du dernier stage
(typiquement /32). Ils n'ont **pas de CLS token** et utilisent un
pooling global pour la classification.

Pour respecter le contrat ``(B, 1 + N, D)`` du pipeline, on synthétise
un CLS = moyenne globale des patches. La grille doit être carrée
(``sqrt(N)`` entier) — vérifié par ``_self_test`` au load.
"""
from __future__ import annotations

import torch

from cd_bench.models.encoders.base import FeatureEncoder, get_hidden_size


class HierSeqLikeEncoder(FeatureEncoder):
    def _infer_shape(self, img_size: int) -> tuple[int, int]:
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, img_size, img_size, device=self.device)
            out = self.model(dummy).last_hidden_state
        if out.ndim != 3:
            raise RuntimeError(
                f"{type(self).__name__}: last_hidden_state ndim={out.ndim}, "
                f"attendu 3 — ce modèle n'est probablement pas hiérarchique-séquentiel."
            )
        n_patches = int(out.shape[1])
        # Le dernier stage Swin a souvent C != model.config.hidden_size
        # (qui peut référer à embed_dim du premier stage). On lit la vraie
        # dim de sortie sur le tenseur.
        dim = int(out.shape[-1])
        # garde-fou : si get_hidden_size diverge, on fait confiance au tensor.
        try:
            cfg_dim = get_hidden_size(self.model.config)
            if cfg_dim != dim:
                pass  # silencieux : la dim runtime est l'autorité.
        except AttributeError:
            pass
        return dim, 1 + n_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.model(pixel_values).last_hidden_state   # (B, N, C)
        cls = patches.mean(dim=1, keepdim=True)                # (B, 1, C)
        return torch.cat([cls, patches], dim=1)                # (B, 1+N, C)
