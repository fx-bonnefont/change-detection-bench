import torch
from sklearn import metrics

def get_classif_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict: 
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    classif_metrics = {
        "accuracy": metrics.accuracy_score(preds_np, targets_np),
        "recall": metrics.recall_score(preds_np, targets_np),
        "precision": metrics.precision_score(preds_np, targets_np),
        "f1": metrics.f1_score(preds_np, targets_np),
    }
    
    return classif_metrics