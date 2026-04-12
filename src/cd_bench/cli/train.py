"""Sous-commande ``cdbench train`` : entraînement SCD sur features pré-extraites."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import typer
from torch.utils.data import DataLoader

from cd_bench.cli.extract import extract as run_extract
from cd_bench.config import MLFLOW_TRACKING_URI, RAW_DATA_DIR, dat_path, metadata_path
from cd_bench.data.feature_dataset import FeatureDataset
from cd_bench.data.mask_mapping import N_CLASSES
from cd_bench.data.splits import SPLIT_FRACTIONS, SPLIT_SEED, split_summary, stratified_split
from cd_bench.models.decoders import DECODERS, get_decoder
from cd_bench.models.encoders import ENCODERS
from cd_bench.training.hparams_store import get_loss_kwargs, get_lr
from cd_bench.training.trainer import evaluate_loader, train as run_training
from cd_bench.utils.device import get_device
from cd_bench.utils.io import load_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cd_bench.train")


def train(
    encoder: str = typer.Option("dinov3-small", "--encoder", "-e"),
    decoder: str = typer.Option("baseline-conv", "--decoder", "-d"),
    batch_size: int = typer.Option(36, "--batch-size", "-b"),
    epochs: int = typer.Option(30, "--epochs"),
    lr: float | None = typer.Option(None, "--lr", help="Learning rate (prioritaire sur le store). Défaut: store ou 1e-4."),
    num_workers: int = typer.Option(2, "--num-workers", "-j"),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Cible les features extraites avec ce même --limit (suffixe -limitN). Smoke testing.",
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint", "-c",
        help="Chemin vers un best.pt pour reprendre l'entraînement.",
    ),
):
    if encoder not in ENCODERS:
        raise typer.BadParameter(f"encoder inconnu. Disponibles : {sorted(ENCODERS)}")
    if decoder not in DECODERS:
        raise typer.BadParameter(f"decoder inconnu. Disponibles : {sorted(DECODERS)}")

    logger.info("encoder=%s decoder=%s", encoder, decoder)
    device = get_device()
    logger.info("device=%s", device)

    if not dat_path(encoder, limit).exists():
        logger.info("Features absentes pour %s (limit=%s) — extraction automatique.", encoder, limit)
        run_extract(encoder=encoder, data_path=str(RAW_DATA_DIR), batch_size=64, num_workers=num_workers, limit=limit)

    meta = load_metadata(metadata_path(encoder, limit))
    model = get_decoder(decoder)(
        d_model=meta["dim"], out_size=meta.get("img_size") or 512,
    ).to(device)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        logger.info("checkpoint loaded from %s (epoch=%d, %s=%.4f)",
                     checkpoint, ckpt["epoch"], ckpt["metric_name"], ckpt["metric_value"])

    splits = stratified_split(meta["items"])
    logger.info("split summary: %s", split_summary(meta["items"], splits))
    train_dataset = FeatureDataset(encoder, indices=splits["train"], load_masks=True, limit=limit)
    valid_dataset = FeatureDataset(encoder, indices=splits["val"], load_masks=True, limit=limit)
    test_dataset = FeatureDataset(encoder, indices=splits["test"], load_masks=True, limit=limit)
    tuned_lr = get_lr(encoder, decoder)
    if lr is not None:
        effective_lr = lr
        lr_source = "cli"
        logger.info("lr: CLI override -> %.2e", effective_lr)
    elif tuned_lr is not None:
        effective_lr = tuned_lr
        lr_source = "store"
        logger.info("lr: loaded from hparams store -> %.2e", effective_lr)
    else:
        effective_lr = 1e-4
        lr_source = "default"
        logger.info("lr: using fallback default -> %.2e", effective_lr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)

    logger.info("train=%d, valid=%d, test=%d samples", len(train_dataset), len(valid_dataset), len(test_dataset))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"CD-BENCH_{decoder}")

    with mlflow.start_run(run_name=encoder):
        mlflow.log_params(
            {
                "decoder": decoder,
                "encoder": encoder,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": effective_lr,
                "lr_source": lr_source,
                "limit": limit,
                "optimizer": "AdamW",
                "split_seed": SPLIT_SEED,
                "split_fractions": str(SPLIT_FRACTIONS),
                "n_classes": N_CLASSES,
            }
        )

        loss_kwargs = get_loss_kwargs(encoder, decoder)
        if loss_kwargs is None:
            logger.warning(
                "loss hparams: no entry in configs/loss_hparams.json. "
                "Using losses.py defaults. Run: cdbench tune -e %s -d %s",
                encoder, decoder,
            )
            mlflow.log_param("loss_hparams_source", "defaults_fallback")
        else:
            logger.info("loss hparams: loaded from store -> %s", loss_kwargs)
            mlflow.log_param("loss_hparams_source", "store")
            mlflow.log_params({f"loss_{k}": v for k, v in loss_kwargs.items()})

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        active_run = mlflow.active_run()
        assert active_run is not None
        run_id = active_run.info.run_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{ts}_{decoder}_{encoder}_{run_id[:8]}.log"
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        logger.info("python logs -> %s", log_file)
        interrupted = False
        try:
            run_training(
                model, train_loader, valid_loader, epochs=epochs, device=device,
                optimizer=optimizer, loss_kwargs=loss_kwargs,
            )
            logger.info("Evaluating on test set...")
            test_results = evaluate_loader(model, test_loader, device=device)
            for k, v in test_results["metrics"].items():
                mlflow.log_metric(f"test_{k}", v)
            logger.info("test metrics: %s", test_results["metrics"])
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("training interrupted by user (KeyboardInterrupt)")
        finally:
            file_handler.flush()
            root_logger.removeHandler(file_handler)
            file_handler.close()
            try:
                mlflow.log_artifact(str(log_file), artifact_path="logs")
                logger.info("log file uploaded to MLflow as logs/%s", log_file.name)
            except Exception as exc:
                logger.warning("Impossible d'uploader le log vers MLflow: %s (fichier local: %s)", exc, log_file)
            if interrupted:
                raise KeyboardInterrupt

    typer.echo(f"Done. Results logged to MLflow: {decoder}")
