"""Chargement d'un modèle entraîné et inférence SCD sur des images brutes."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from cd_bench.data.mask_mapping import N_CLASSES
from cd_bench.models.cd_model import CDModel
from cd_bench.models.decoders import get_decoder
from cd_bench.models.encoders import get_encoder
from cd_bench.utils.device import get_device


def load_model(
    encoder_name: str,
    decoder_name: str,
    checkpoint_path: Path,
    img_size: int = 512,
) -> CDModel:
    """Charge un encoder + decoder depuis un checkpoint et retourne un ``CDModel`` prêt à inférer."""
    device = get_device()
    enc = get_encoder(encoder_name)
    enc.load(img_size)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    dec = get_decoder(decoder_name)(d_model=enc.dim, out_size=img_size)
    dec.load_state_dict(state_dict)

    model = CDModel(enc, dec).to(device)
    model.eval()
    return model


def predict_pair(
    model: CDModel,
    img_2018: Image.Image,
    img_2019: Image.Image,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inférence SCD sur une paire d'images PIL.

    Retourne ``(sem_t1, sem_t2, change_mask)`` de forme ``(H, W)`` chacun, en CPU.
    - ``sem_t1`` / ``sem_t2`` : classes sémantiques prédites (argmax) int64
    - ``change_mask`` : bool, ``True`` là où ``sem_t1 != sem_t2``
    """
    processor = model.encoder.processor
    device = model.encoder.device

    pv1 = processor(img_2018, return_tensors="pt")["pixel_values"].to(device)
    pv2 = processor(img_2019, return_tensors="pt")["pixel_values"].to(device)

    K = N_CLASSES + 1
    with torch.inference_mode():
        logits = model(pv1, pv2)  # (1, 2K, H, W)
        pred_t1 = logits[0, :K].argmax(dim=0).cpu()  # (H, W)
        pred_t2 = logits[0, K:].argmax(dim=0).cpu()   # (H, W)

    change_mask = pred_t1 != pred_t2
    return pred_t1, pred_t2, change_mask
