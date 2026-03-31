import torch
import mlflow
import numpy as np
from tqdm import trange
from sklearn.metrics import roc_curve
from src.training.metrics import get_classif_metrics


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = (tpr - fpr).argmax()
    return float(thresholds[best_idx])


def log_results(results: dict):
    mlflow.log_metric("threshold", results["threshold"])
    for metric_name, value in results["metrics"].items():
        mlflow.log_metric(metric_name, value)


def evaluate_baseline(model, dataset, batch_size=256) -> dict:
    model.eval()
    n = len(dataset)
    labels = dataset.binary_labels.numpy()
    all_scores = []

    with torch.inference_mode():
        for start in trange(0, n, batch_size, desc="Evaluate"):
            end = min(start + batch_size, n)
            t1 = torch.from_numpy(np.array(dataset.features_2018[start:end]))
            t2 = torch.from_numpy(np.array(dataset.features_2019[start:end]))
            scores = model(t1, t2)
            all_scores.append(scores)

    scores = torch.cat(all_scores).numpy()
    threshold = find_best_threshold(1 - scores, labels)
    preds = torch.from_numpy((scores < (1 - threshold)).astype(int))
    labels_t = torch.from_numpy(labels.astype(int))

    return {
        "threshold": threshold,
        "metrics": get_classif_metrics(preds, labels_t),
    }


def train(model, train_loader, valid_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            t1, t2, binary_label = batch[0], batch[1], batch[2]
            optimizer.zero_grad()
            output = model(t1, t2)
            loss = loss_fn(output, binary_label.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    return {"train_loss": avg_loss}
