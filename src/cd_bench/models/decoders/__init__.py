"""Registry des têtes de décodage (siamoise, middle-fusion).

Toutes les têtes consomment ``(B, N_tokens, D_model)`` et produisent des
logits ``(B, 2K, out_size, out_size)`` (K classes par date pour le SCD).
Ajouter une nouvelle tête = écrire un module + l'enregistrer dans ``DECODERS``.
"""
from cd_bench.models.decoders.baseline_conv import BaselineConvHead
from cd_bench.models.decoders.query_decoder import ChangeQueryDecoder


DECODERS = {
    "baseline-conv": BaselineConvHead,
    "query-decoder": ChangeQueryDecoder,
}


def get_decoder(name: str):
    if name not in DECODERS:
        raise KeyError(f"Unknown decoder '{name}'. Available: {sorted(DECODERS)}")
    return DECODERS[name]


__all__ = ["BaselineConvHead", "ChangeQueryDecoder", "DECODERS", "get_decoder"]
