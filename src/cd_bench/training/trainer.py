import logging
import os
import tempfile
import time

import mlflow
import torch
from tqdm import tqdm

from cd_bench.data.mask_mapping import IGNORE_INDEX, N_CLASSES
from cd_bench.training.losses import SCDLoss

logger = logging.getLogger(__name__)


def _save_best_checkpoint(model, epoch: int, metric_value: float, metric_name: str) -> None:
    """Sérialise le state_dict et l'enregistre comme artefact MLflow."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "best.pt")
        logger.info("epoch %d: saving checkpoint to %s", epoch + 1, path)
        torch.save(
            {
                "epoch": epoch,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "state_dict": model.state_dict(),
            },
            path,
        )
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info("epoch %d: uploading checkpoint to MLflow (%.1f MB)…", epoch + 1, size_mb)
        t0 = time.perf_counter()
        mlflow.log_artifact(path, artifact_path="checkpoints")
        logger.info("epoch %d: checkpoint uploaded in %.1fs", epoch + 1, time.perf_counter() - t0)
    logger.info("epoch %d: new best %s=%.4f", epoch + 1, metric_name, metric_value)


def train(
    model,
    train_loader,
    valid_loader,
    epochs,
    device=None,
    optimizer=None,
    best_metric_name: str = "mean_iou",
    loss_kwargs: dict | None = None,
    save_checkpoints: bool = True,
):
    """Boucle d'entraînement SCD.

    À chaque epoch :
      - entraînement sur ``train_loader``
      - évaluation sur ``valid_loader`` (si fourni)
      - sauvegarde du meilleur ``state_dict`` comme artefact MLflow
    """
    device = device or next(model.parameters()).device
    model.to(device)
    if optimizer is None:
        raise ValueError("optimizer must be provided")

    n_classes = N_CLASSES + 1  # 0..N_CLASSES
    loss_fn = SCDLoss(n_classes=n_classes, **(loss_kwargs or {}))

    n_batches = len(train_loader)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("train: device=%s, epochs=%d, batches/epoch=%d, trainable_params=%d",
                device, epochs, n_batches, n_params)

    avg_loss = float("nan")
    best_metric = -float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar, start=1):
            t1, t2 = batch[0].to(device), batch[1].to(device)
            sem_t1 = batch[2].to(device)
            sem_t2 = batch[3].to(device)
            valid = batch[4].to(device).float()

            optimizer.zero_grad()
            logits = model(t1, t2)
            if logits.shape[-2:] != sem_t1.shape[-2:]:
                raise ValueError(
                    f"Shape mismatch logits {tuple(logits.shape)} vs target {tuple(sem_t1.shape)}"
                )
            loss = loss_fn(logits, sem_t1, sem_t2, valid)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{epoch_loss / step:.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_dt = time.perf_counter() - epoch_start
        logger.info("epoch %d/%d done in %.1fs (avg_loss=%.4f)",
                    epoch + 1, epochs, epoch_dt, avg_loss)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if valid_loader is not None:
            val_results = evaluate_loader(model, valid_loader, device=device)
            for k, v in val_results["metrics"].items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)
            current = val_results["metrics"].get(best_metric_name)
            if current is not None and current > best_metric:
                best_metric = current
                if save_checkpoints:
                    _save_best_checkpoint(model, epoch, current, best_metric_name)

    # Retourner aussi les dernières métriques de validation
    last_val = val_results["metrics"] if valid_loader is not None else {}
    return {"train_loss": avg_loss, f"best_{best_metric_name}": best_metric, **last_val}


def evaluate_loader(
    model,
    loader,
    device=None,
) -> dict:
    """Évaluation SCD : mIoU sémantique par date + BCD dérivé.

    BCD dérivé = ``argmax(pred_t1) != argmax(pred_t2)`` vs
    ``sem_t1 != sem_t2`` sur les pixels valides (hors ignore_index des deux côtés).
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    n_classes = N_CLASSES + 1
    tp_t1 = torch.zeros(n_classes, device=device)
    fp_t1 = torch.zeros(n_classes, device=device)
    fn_t1 = torch.zeros(n_classes, device=device)
    tp_t2 = torch.zeros(n_classes, device=device)
    fp_t2 = torch.zeros(n_classes, device=device)
    fn_t2 = torch.zeros(n_classes, device=device)
    bcd_tp = 0.0
    bcd_fp = 0.0
    bcd_fn = 0.0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="EvaluateSCD"):
            t1, t2 = batch[0].to(device), batch[1].to(device)
            sem_t1 = batch[2].to(device)
            sem_t2 = batch[3].to(device)
            valid = batch[4].to(device).float()

            logits = model(t1, t2)  # (B, 2K, H, W)
            K = n_classes
            pred_t1 = logits[:, :K].argmax(dim=1)  # (B, H, W)
            pred_t2 = logits[:, K:].argmax(dim=1)

            valid_bool = valid.bool()

            for c in range(n_classes):
                if c == IGNORE_INDEX:
                    continue
                p1 = (pred_t1 == c) & valid_bool
                g1 = (sem_t1 == c) & valid_bool
                tp_t1[c] += (p1 & g1).sum()
                fp_t1[c] += (p1 & ~g1).sum()
                fn_t1[c] += (~p1 & g1).sum()

                p2 = (pred_t2 == c) & valid_bool
                g2 = (sem_t2 == c) & valid_bool
                tp_t2[c] += (p2 & g2).sum()
                fp_t2[c] += (p2 & ~g2).sum()
                fn_t2[c] += (~p2 & g2).sum()

            sem_valid = valid_bool & (sem_t1 != IGNORE_INDEX) & (sem_t2 != IGNORE_INDEX)
            gt_change = (sem_t1 != sem_t2) & sem_valid
            pred_change = (pred_t1 != pred_t2) & sem_valid
            bcd_tp += (pred_change & gt_change).sum().item()
            bcd_fp += (pred_change & ~gt_change).sum().item()
            bcd_fn += (~pred_change & gt_change).sum().item()

    if was_training:
        model.train()

    eps = 1e-7
    ious_t1, ious_t2 = [], []
    for c in range(n_classes):
        if c == IGNORE_INDEX:
            continue
        iou1 = tp_t1[c].item() / (tp_t1[c].item() + fp_t1[c].item() + fn_t1[c].item() + eps)
        iou2 = tp_t2[c].item() / (tp_t2[c].item() + fp_t2[c].item() + fn_t2[c].item() + eps)
        ious_t1.append(iou1)
        ious_t2.append(iou2)

    miou_t1 = sum(ious_t1) / len(ious_t1) if ious_t1 else 0.0
    miou_t2 = sum(ious_t2) / len(ious_t2) if ious_t2 else 0.0
    mean_iou = (miou_t1 + miou_t2) / 2.0
    bcd_iou = bcd_tp / (bcd_tp + bcd_fp + bcd_fn + eps)

    return {
        "metrics": {
            "miou_t1": miou_t1,
            "miou_t2": miou_t2,
            "mean_iou": mean_iou,
            "bcd_iou": bcd_iou,
        }
    }
