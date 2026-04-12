"""Encoder ViT HuggingFace standard : sortie ``CLS + patches``.

Couvre la majorité des ViT pré-entraînés sans tokens spéciaux : ViT,
BEiT, EVA, MAE, CLIP-ViT, SigLIP, etc. ``last_hidden_state`` est
renvoyé tel quel — son layout est déjà ``(B, 1 + N_patches, D)``.
"""
from __future__ import annotations

import logging

import torch

from cd_bench.models.encoders.base import FeatureEncoder, get_hidden_size

logger = logging.getLogger(__name__)


class ViTLikeEncoder(FeatureEncoder):
    def _infer_shape(self, img_size: int) -> tuple[int, int]:
        self._warn_if_resolution_mismatch(img_size)
        dim = get_hidden_size(self.model.config)
        patch_size = getattr(self.model.config, "patch_size", None)
        if patch_size is None:
            with torch.inference_mode():
                dummy = torch.zeros(1, 3, img_size, img_size, device=self.device)
                out = self._forward_model(dummy).last_hidden_state
            return dim, int(out.shape[1])
        n_patches = (img_size // patch_size) ** 2
        return dim, 1 + n_patches

    def _warn_if_resolution_mismatch(self, img_size: int) -> None:
        """Prévient si les positional embeddings vont être interpolés.

        ViT vanilla (HF) n'accepte que sa résolution native sauf si on
        passe ``interpolate_pos_encoding=True`` — ce qu'on fait
        systématiquement dans ``_forward_model``. L'interpolation marche
        mathématiquement mais **dégrade la qualité** des features par
        rapport à un checkpoint pré-entraîné nativement à la bonne
        résolution. On loggue donc un warning explicite pour que
        l'utilisateur soit conscient du compromis.
        """
        native = getattr(self.model.config, "image_size", None)
        if native is None:
            return
        if isinstance(native, (list, tuple)):
            native_size = int(native[0])
        else:
            native_size = int(native)
        if native_size != img_size:
            logger.warning(
                "%s (%s): images %dx%d ≠ résolution native du checkpoint (%dx%d). "
                "Les positional embeddings vont être interpolés bilinéairement à "
                "chaque forward (interpolate_pos_encoding=True). Les features sont "
                "valides mais de qualité dégradée vs un checkpoint pré-entraîné "
                "nativement à %dx%d — envisage `cdbench search` pour trouver une "
                "variante haute résolution.",
                type(self).__name__, self.name, img_size, img_size,
                native_size, native_size, img_size,
            )

    def _forward_model(self, pixel_values: torch.Tensor):
        """Forward le modèle HF en demandant l'interpolation des positional
        embeddings quand l'input n'est pas à la résolution native du
        pré-entraînement (ex: ViT-base 224 utilisé en 512)."""
        try:
            return self.model(pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            return self.model(pixel_values)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._forward_model(pixel_values).last_hidden_state
