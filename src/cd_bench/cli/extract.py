"""Sous-commande ``cdbench extract`` : extraction des features encoder vers memmap.

Format de sortie (``metadata_*.json``) — ``format_version: 2`` :

    {
      "format_version": 2,
      "encoder": ..., "hf_id": ..., "dim": ..., "n_tokens": ..., "img_size": ...,
      "limit": ...,
      "sections": {
        "features_2018": {"offset": 0,    "shape": [N, n_tokens, dim], "dtype": "float32"},
        "features_2019": {"offset": OFF1, "shape": [N, n_tokens, dim], "dtype": "float32"}
      },
      "items": [
        {"id": ..., "mask_path": ..., "original_split": "train"|"valid",
         "changed_ratio": float, "valid_ratio": float},
        ...
      ]
    }

Tous les tiles HI-UCD ayant des masques (officiels ``train`` + ``valid``)
sont concaténés dans un pool unique. Le découpage train/val/test est
ensuite fait au runtime via :mod:`cd_bench.data.splits`.
"""
from __future__ import annotations

import numpy as np
import torch
import typer
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from cd_bench.config import RAW_DATA_DIR, dat_path, encoder_dir, metadata_path
from cd_bench.data.paths import get_paths
from cd_bench.data.raw_dataset import ImageDataset
from cd_bench.models.encoders import ENCODERS, get_encoder
from cd_bench.utils.io import create_memmap, flush, save_metadata, write_chunk

FORMAT_VERSION = 2


def _build_tile_list(paths: dict) -> list[dict]:
    """Concatène train + valid en une liste plate alignée 2018/2019/mask."""
    tiles: list[dict] = []
    for split in ("train", "valid"):
        part = paths.get(split)
        if not part:
            continue
        p2018 = part.get("2018", [])
        p2019 = part.get("2019", [])
        masks = part.get("mask", [])
        if not (len(p2018) == len(p2019) == len(masks)):
            raise ValueError(
                f"Mismatch lengths in split={split}: "
                f"2018={len(p2018)}, 2019={len(p2019)}, mask={len(masks)}"
            )
        for a, b, m in zip(p2018, p2019, masks):
            tiles.append(
                {
                    "id": a.stem,
                    "p2018": a,
                    "p2019": b,
                    "mask": m,
                    "original_split": split,
                }
            )
    return tiles


def _compute_mask_ratios(tiles: list[dict]) -> None:
    """Renseigne ``changed_ratio`` et ``valid_ratio`` pour chaque tile."""
    for t in tqdm(tiles, desc="Computing mask ratios"):
        with Image.open(t["mask"]) as im:
            change_ch = np.array(im, dtype=np.uint8)[..., 2]
        n_pixels = float(change_ch.size)
        valid = (change_ch != 0)
        v = float(valid.sum())
        c = float((change_ch == 2).sum())
        t["valid_ratio"] = v / n_pixels if n_pixels > 0 else 0.0
        t["changed_ratio"] = c / v if v > 0 else 0.0


def _extract_period(
    image_paths: list,
    encoder,
    fp,
    byte_offset: int,
    batch_size: int,
    num_workers: int,
    desc: str,
) -> int:
    dataset = ImageDataset(image_paths, encoder.processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc):
            pixel_values = (
                batch["pixel_values"].to(encoder.device)
                if isinstance(batch, dict)
                else batch.to(encoder.device)
            )
            features = encoder.forward(pixel_values).cpu().numpy().astype(np.float32)
            byte_offset = write_chunk(fp, features, byte_offset)
    flush(fp)
    return byte_offset


def extract(
    encoder: str = typer.Option("dinov3-small", "--encoder", "-e", help="Nom de l'encoder dans le registry."),
    data_path: str = typer.Option(str(RAW_DATA_DIR), "--data-path", help="Racine des images brutes HI-UCD."),
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    num_workers: int = typer.Option(2, "--num-workers", "-j"),
    limit: int | None = typer.Option(None, "--limit", help="Smoke test : N premières images, stockées sous <encoder>-limit<N>."),
):
    if encoder not in ENCODERS:
        raise typer.BadParameter(f"encoder inconnu. Disponibles : {sorted(ENCODERS)}")

    paths = get_paths(data_path, limit)
    with Image.open(paths["train"]["2018"][0]) as img:
        img_size = img.size[0]

    if dat_path(encoder, limit).exists():
        typer.echo(f"Features {encoder} (limit={limit}) already exist, skipping.")
        return

    # 1. Construire la liste plate de tiles (train + valid HI-UCD).
    tiles = _build_tile_list(paths)
    typer.echo(f"Total tiles to extract: {len(tiles)}")

    # 2. Calculer changed_ratio / valid_ratio en amont (rapide, masques petits).
    _compute_mask_ratios(tiles)

    # 3. Charger l'encoder.
    enc = get_encoder(encoder)
    enc.load(img_size)

    out_dir = encoder_dir(encoder, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = create_memmap(dat_path(encoder, limit))

    n_total = len(tiles)
    sections: dict[str, dict] = {}

    # 4. Extraction features 2018 puis 2019, **dans le même ordre** que ``tiles``.
    byte_offset = 0
    sections["features_2018"] = {
        "offset": byte_offset,
        "shape": [n_total, enc.n_tokens, enc.dim],
        "dtype": "float32",
    }
    byte_offset = _extract_period(
        [t["p2018"] for t in tiles], enc, fp, byte_offset, batch_size, num_workers,
        desc="Extracting 2018",
    )

    sections["features_2019"] = {
        "offset": byte_offset,
        "shape": [n_total, enc.n_tokens, enc.dim],
        "dtype": "float32",
    }
    byte_offset = _extract_period(
        [t["p2019"] for t in tiles], enc, fp, byte_offset, batch_size, num_workers,
        desc="Extracting 2019",
    )

    # 5. Sérialiser les items en chemins relatifs (portable entre machines).
    root = RAW_DATA_DIR
    items = []
    for t in tiles:
        try:
            mask_rel = str(t["mask"].relative_to(root))
        except ValueError:
            mask_rel = str(t["mask"])
        items.append(
            {
                "id": t["id"],
                "mask_path": mask_rel,
                "original_split": t["original_split"],
                "changed_ratio": float(t["changed_ratio"]),
                "valid_ratio": float(t["valid_ratio"]),
            }
        )

    metadata = {
        "format_version": FORMAT_VERSION,
        "encoder": encoder,
        "hf_id": enc.hf_id,
        "dim": enc.dim,
        "n_tokens": enc.n_tokens,
        "img_size": img_size,
        "limit": limit,
        "sections": sections,
        "items": items,
    }
    save_metadata(metadata_path(encoder, limit), metadata)
    typer.echo(f"Completed. Data: {dat_path(encoder, limit)}")
