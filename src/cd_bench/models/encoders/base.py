"""Interface commune des encoders et auto-test au chargement.

Contrat (toutes sous-classes doivent le respecter) :
    - ``forward(pixel_values) -> (B, n_tokens, dim)``
    - ``n_tokens == 1 + H*W`` avec ``H*W`` carré parfait : index 0 = CLS
      (synthétique si nécessaire), indices 1.. = patches sur grille carrée.

Le décodeur (``RichFusion`` + reshape) repose sur ce contrat. Tout
encoder qui ne peut pas le tenir doit lever ``NotImplementedError``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from transformers import AutoImageProcessor, AutoModel

from cd_bench.utils.device import get_device


def get_hidden_size(cfg) -> int:
    """Lookup tolérant aux conventions HF (`hidden_size`, `embed_dim`, …)."""
    for attr in ("hidden_size", "embed_dim", "hidden_dim"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    if hasattr(cfg, "hidden_sizes"):  # ConvNeXt etc. : liste, dernier stage
        return int(cfg.hidden_sizes[-1])
    raise AttributeError(f"no hidden_size attribute on {type(cfg).__name__}")


class FeatureEncoder(ABC):
    name: str
    hf_id: str

    def __init__(self, name: str, hf_id: str):
        self.name = name
        self.hf_id = hf_id
        self.processor = None
        self.model = None
        self.device = None
        self.dim: int | None = None
        self.n_tokens: int | None = None

    def load(self, img_size: int) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.hf_id)
        # Les images HI-UCD sont déjà à la résolution cible : on coupe le
        # resize/center-crop du processor pour ne garder que la
        # normalisation (ImageNet/CLIP/SigLIP stats selon le modèle).
        # Évite les conflits de convention entre processors HF
        # ({"height","width"} pour ViT/DINOv3 vs {"shortest_edge"} pour
        # ConvNext, etc.).
        for attr in ("do_resize", "do_center_crop"):
            if hasattr(self.processor, attr):
                setattr(self.processor, attr, False)
        self.device = get_device()
        self.model = AutoModel.from_pretrained(self.hf_id).to(self.device).eval()
        self.dim, self.n_tokens = self._infer_shape(img_size)
        self._self_test(img_size)

    def _self_test(self, img_size: int) -> None:
        """Forward d'un tenseur factice et vérifie le contrat de sortie.

        Attrape ~80 % des bugs (mauvais slicing de tokens spéciaux, dim
        incorrect, grille non carrée, modèle hiérarchique non géré, …)
        dès le ``load()`` plutôt qu'à mi-entraînement.
        """
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, img_size, img_size, device=self.device)
            try:
                out = self.forward(dummy)
            except Exception as e:
                raise RuntimeError(
                    f"{type(self).__name__}({self.name}): forward a échoué sur "
                    f"un tenseur factice ({img_size}x{img_size}). Cause : {e}"
                ) from e

        if out.ndim != 3 or tuple(out.shape) != (1, self.n_tokens, self.dim):
            raise RuntimeError(
                f"{type(self).__name__}({self.name}): forward a renvoyé "
                f"shape={tuple(out.shape)}, attendu (1, {self.n_tokens}, {self.dim}). "
                f"Probable mismatch entre _infer_shape et forward "
                f"(tokens spéciaux mal slicés ?)."
            )

        n_patches = self.n_tokens - 1
        side = round(n_patches ** 0.5)
        if side * side != n_patches:
            raise RuntimeError(
                f"{type(self).__name__}({self.name}): n_patches={n_patches} "
                f"n'est pas un carré parfait — les décodeurs (RichFusion + "
                f"reshape grille) vont casser. Modèle hiérarchique non supporté ?"
            )

    @abstractmethod
    def _infer_shape(self, img_size: int) -> tuple[int, int]:
        """Retourne ``(dim, n_tokens)`` attendus en sortie de ``forward``."""

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``pixel_values`` -> ``(B, n_tokens, dim)`` (CLS à l'index 0, patches ensuite)."""
