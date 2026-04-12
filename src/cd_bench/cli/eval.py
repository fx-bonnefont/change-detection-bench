"""Sous-commande ``cdbench eval`` : évaluation d'un checkpoint sur val ou test.

Charge un modèle (décodeur seul, mode frozen), évalue sur un split,
et affiche les métriques SCD. Aucun entraînement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from torch.utils.data import DataLoader

from cd_bench.config import MLFLOW_TRACKING_URI, dat_path, metadata_path
from cd_bench.data.feature_dataset import FeatureDataset
from cd_bench.data.splits import stratified_split
from cd_bench.inference.predict import load_model
from cd_bench.training.trainer import evaluate_loader
from cd_bench.utils.device import get_device
from cd_bench.utils.io import load_metadata


def _resolve_checkpoint(checkpoint: str | None, run_id: str | None) -> str:
    if checkpoint is not None:
        p = Path(checkpoint)
        if not p.exists():
            raise typer.BadParameter(f"Checkpoint introuvable : {p}")
        return str(p)
    if run_id is not None:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        return client.download_artifacts(run_id, "checkpoints/best.pt")
    raise typer.BadParameter("Il faut fournir --checkpoint ou --run-id.")


def eval(
    encoder: str = typer.Option("dinov3-small", "--encoder", "-e"),
    decoder: str = typer.Option("baseline-conv", "--decoder", "-d"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Chemin local vers best.pt."),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="MLflow run_id (télécharge checkpoints/best.pt)."),
    split: str = typer.Option("val", "--split", help="Split à évaluer : val ou test."),
    batch_size: int = typer.Option(32, "--batch-size", "-b"),
    num_workers: int = typer.Option(2, "--num-workers", "-j"),
    limit: Optional[int] = typer.Option(None, "--limit"),
):
    """Évalue un checkpoint sur val ou test (sans entraînement)."""
    if split not in ("val", "test"):
        raise typer.BadParameter("--split doit être 'val' ou 'test'.")

    if not dat_path(encoder, limit).exists():
        raise typer.BadParameter(f"Features absentes pour {encoder} (limit={limit}). Lance `cdbench extract` d'abord.")

    meta = load_metadata(metadata_path(encoder, limit))
    splits = stratified_split(meta["items"])
    indices = splits[split]
    typer.echo(f"Split '{split}' : {len(indices)} tiles.")

    dataset = FeatureDataset(encoder, indices=indices, load_masks=True, limit=limit)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    ckpt_path = _resolve_checkpoint(checkpoint, run_id)
    typer.echo(f"Chargement du modèle {encoder} + {decoder} depuis {ckpt_path}...")
    model = load_model(encoder, decoder, ckpt_path)
    device = get_device()

    typer.echo("Évaluation SCD...")
    results = evaluate_loader(model.decoder, loader, device=device)

    typer.echo("")
    typer.echo(f"{'Metric':<20} {'Value':>10}")
    typer.echo("-" * 32)
    for k, v in results["metrics"].items():
        typer.echo(f"{k:<20} {v:>10.4f}")
