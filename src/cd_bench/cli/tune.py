"""Sous-commande ``cdbench tune`` : recherche d'hyperparamètres via Optuna.

Optimise ``lambda_bcd``, ``bcd_alpha``, ``bcd_gamma`` et ``lr`` de
:class:`SCDLoss` en maximisant ``0.75*bcd_iou + 0.25*mean_iou`` sur le
split ``val``. Le split ``test`` n'est **jamais** vu pendant le tuning.

Chaque trial = 1 run MLflow nested sous une parent run "sweep".
"""
from __future__ import annotations

import logging

import mlflow
import optuna
import torch
import typer
from torch.utils.data import DataLoader

from cd_bench.config import MLFLOW_TRACKING_URI, dat_path, metadata_path
from cd_bench.data.feature_dataset import FeatureDataset
from cd_bench.data.splits import SPLIT_FRACTIONS, SPLIT_SEED, split_summary, stratified_split
from cd_bench.models.decoders import DECODERS, get_decoder
from cd_bench.models.encoders import ENCODERS
from cd_bench.training.hparams_store import upsert_if_better
from cd_bench.training.trainer import train as run_training
from cd_bench.utils.device import get_device
from cd_bench.utils.io import load_metadata

logger = logging.getLogger("cd_bench.tune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _build_decoder(decoder_name: str, meta: dict):
    return get_decoder(decoder_name)(d_model=meta["dim"], out_size=meta["img_size"])


def tune(
    encoder: str = typer.Option("dinov3-small", "--encoder", "-e"),
    decoder: str = typer.Option("baseline-conv", "--decoder", "-d"),
    n_trials: int = typer.Option(12, "--n-trials"),
    epochs: int = typer.Option(6, "--epochs", help="Epochs courts par trial."),
    batch_size: int = typer.Option(32, "--batch-size", "-b"),
    lr: float = typer.Option(1e-4, "--lr"),
    num_workers: int = typer.Option(2, "--num-workers", "-j"),
    limit: int | None = typer.Option(None, "--limit"),
    study_name: str = typer.Option("scd-hp-sweep", "--study-name"),
):
    if encoder not in ENCODERS:
        raise typer.BadParameter(f"encoder inconnu. Disponibles : {sorted(ENCODERS)}")
    if decoder not in DECODERS:
        raise typer.BadParameter(f"decoder inconnu. Disponibles : {sorted(DECODERS)}")

    if not dat_path(encoder, limit).exists():
        raise typer.BadParameter(
            f"Features absentes pour {encoder} (limit={limit}). Lance `cdbench extract` d'abord."
        )

    device = get_device()
    meta = load_metadata(metadata_path(encoder, limit))
    splits = stratified_split(meta["items"])
    logger.info("split summary: %s", split_summary(meta["items"], splits))

    train_ds = FeatureDataset(encoder, indices=splits["train"], load_masks=True, limit=limit)
    val_ds = FeatureDataset(encoder, indices=splits["val"], load_masks=True, limit=limit)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"CD-BENCH_{decoder}_TUNE")

    def objective(trial: optuna.Trial) -> float:
        loss_kwargs = {
            "lambda_dice": trial.suggest_float("lambda_dice", 0.1, 10.0, log=True),
            "lambda_bcd": trial.suggest_float("lambda_bcd", 5.0, 20.0, log=True),
            "bcd_alpha": trial.suggest_float("bcd_alpha", 0.7, 0.95),
            "bcd_gamma": trial.suggest_float("bcd_gamma", 1.0, 3.0),
        }
        trial_lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(
                {
                    "encoder": encoder,
                    "decoder": decoder,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": trial_lr,
                    "limit": limit,
                    "split_seed": SPLIT_SEED,
                    "split_fractions": str(SPLIT_FRACTIONS),
                    **loss_kwargs,
                }
            )
            model = _build_decoder(decoder, meta).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=trial_lr)
            best = 0.0
            for ep in range(epochs):
                results = run_training(
                    model,
                    train_loader,
                    val_loader,
                    epochs=1,
                    device=device,
                    optimizer=optimizer,
                    loss_kwargs=loss_kwargs,
                    save_checkpoints=False,
                )
                miou = float(results.get("mean_iou", 0.0))
                bcd = float(results.get("bcd_iou", 0.0))
                score = 0.75 * bcd + 0.25 * miou
                best = max(best, score)
                trial.report(score, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mlflow.log_metric("trial_best_score", best)
            return best

    storage = f"sqlite:///optuna-{study_name}.db"
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        study_name=study_name, storage=storage, load_if_exists=True,
    )
    done = len(study.trials)
    remaining = max(0, n_trials - done)
    if done > 0:
        logger.info("Resuming study '%s': %d trials done, %d remaining.", study_name, done, remaining)
    if remaining == 0:
        logger.info("Study already has %d trials (>= n_trials=%d). Syncing best to store.", done, n_trials)
        best_params = dict(study.best_trial.params)
        updated, prev = upsert_if_better(
            encoder=encoder,
            decoder=decoder,
            loss_kwargs=best_params,
            best_score=study.best_value,
            n_trials=done,
            epochs_per_trial=epochs,
        )
        logger.info("Best trial: %s (score=%.4f, store_updated=%s)", study.best_trial.params, study.best_value, updated)
        return

    with mlflow.start_run(run_name=f"sweep_{study_name}"):
        mlflow.log_params(
            {
                "encoder": encoder,
                "decoder": decoder,
                "n_trials": n_trials,
                "epochs_per_trial": epochs,
                "split_seed": SPLIT_SEED,
            }
        )
        study.optimize(objective, n_trials=remaining)

        logger.info("Best trial: %s", study.best_trial.params)
        logger.info("Best score (0.75*bcd + 0.25*miou): %.4f", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})
        mlflow.log_metric("best_score_overall", study.best_value)

        best_params = dict(study.best_trial.params)
        updated, prev = upsert_if_better(
            encoder=encoder,
            decoder=decoder,
            loss_kwargs=best_params,
            best_score=study.best_value,
            n_trials=n_trials,
            epochs_per_trial=epochs,
        )
        mlflow.log_param("hparams_store_updated", updated)
        if prev is not None:
            mlflow.log_metric("hparams_store_previous_best", prev)
