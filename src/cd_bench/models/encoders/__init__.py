"""Registry des encoders disponibles pour l'extraction de features.

Quatre familles couvrent la quasi-totalité de HuggingFace vision :

    - :class:`ViTLikeEncoder`      -> ViT, BEiT, EVA, CLIP-ViT, SigLIP, …
    - :class:`ViTRegLikeEncoder`   -> DINOv2-reg, DINOv3 (drop register tokens)
    - :class:`ConvLikeEncoder`     -> ConvNeXt, ResNet (sortie spatiale 4D)
    - :class:`HierSeqLikeEncoder`  -> Swin, SwinV2, FocalNet (séquence hiérarchique 3D)

Ajouter un encoder = ajouter une entrée ci-dessous. Le ``_self_test``
intégré au ``load()`` valide automatiquement le contrat de sortie
``(B, 1 + N_patches_carré, D)``.
"""
from cd_bench.models.encoders.base import FeatureEncoder
from cd_bench.models.encoders.conv_like import ConvLikeEncoder
from cd_bench.models.encoders.hier_seq_like import HierSeqLikeEncoder
from cd_bench.models.encoders.vit_like import ViTLikeEncoder
from cd_bench.models.encoders.vit_reg_like import ViTRegLikeEncoder


ENCODERS: dict[str, FeatureEncoder] = {
    # DinoV3 family : ViT + 4 register tokens
    "dinov3-small":      ViTRegLikeEncoder("dinov3-small",      "facebook/dinov3-vits16-pretrain-lvd1689m",       num_registers=4),
    "dinov3-small-plus": ViTRegLikeEncoder("dinov3-small-plus", "facebook/dinov3-vits16plus-pretrain-lvd1689m",   num_registers=4),
    "dinov3-base":       ViTRegLikeEncoder("dinov3-base",       "facebook/dinov3-vitb16-pretrain-lvd1689m",       num_registers=4),
    "dinov3-large":      ViTRegLikeEncoder("dinov3-large",      "facebook/dinov3-vitl16-pretrain-lvd1689m",       num_registers=4),
    "dinov3-large-sat":  ViTRegLikeEncoder("dinov3-large-sat",  "facebook/dinov3-vitl16-pretrain-sat493m",        num_registers=4),
    "dinov3-huge-plus":  ViTRegLikeEncoder("dinov3-huge-plus",  "facebook/dinov3-vith16plus-pretrain-lvd1689m",   num_registers=4),
    "dinov3-7b":         ViTRegLikeEncoder("dinov3-7b",         "facebook/dinov3-vit7b16-pretrain-lvd1689m",      num_registers=4),
    "dinov3-7b-sat":     ViTRegLikeEncoder("dinov3-7b-sat",     "facebook/dinov3-vit7b16-pretrain-sat493m",       num_registers=4),

    # ViT standard (CLS + patches)
    "vit-base":      ViTLikeEncoder("vit-base",      "google/vit-base-patch16-224"),

    # Convolutionnel / hiérarchique 4D
    "convnext-base": ConvLikeEncoder("convnext-base", "facebook/convnext-base-224"),

    # Hiérarchique séquentiel (pas de CLS, mean-pool synthétique)
    "swinv2-base":   HierSeqLikeEncoder("swinv2-base", "microsoft/swinv2-base-patch4-window8-256"),
    "focalnet-base": HierSeqLikeEncoder("focalnet-base", "microsoft/focalnet-base"),
}


def get_encoder(name: str) -> FeatureEncoder:
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder '{name}'. Available: {sorted(ENCODERS)}")
    return ENCODERS[name]


__all__ = [
    "FeatureEncoder",
    "ViTLikeEncoder",
    "ViTRegLikeEncoder",
    "ConvLikeEncoder",
    "HierSeqLikeEncoder",
    "ENCODERS",
    "get_encoder",
]
