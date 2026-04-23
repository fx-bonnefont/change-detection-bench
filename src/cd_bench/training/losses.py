"""Pertes pour le Semantic Change Detection (SCD).

Le modèle prédit ``(B, 2K, H, W)`` : les K premiers canaux = carte
sémantique à T1, les K suivants = carte sémantique à T2.

On combine :
  1. CE multi-classe (symétriques sur T1 et T2)
  2. Un terme auxiliaire focal binaire qui force le modèle à détecter les
     zones de changement (là où sem_t1 ≠ sem_t2). Sans ce terme, le gradient
     des ~1% de pixels changed est noyé par les ~99% unchanged et le
     modèle apprend le raccourci "même carte partout".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryChangeLoss(nn.Module):
    """Focal loss binaire sur le signal de changement dérivé des softmax.

    La probabilité de changement est calculée de manière différentiable :
    ``p_change = 1 - sum_c(softmax_t1_c * softmax_t2_c)`` — c'est 0 quand
    les distributions sont identiques et ~1 quand elles divergent.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        probs_t1: torch.Tensor,
        probs_t2: torch.Tensor,
        change_gt: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """probs_t1/t2: (B, K, H, W) softmax, change_gt: (B, H, W) float {0,1}, valid: (B, H, W) float."""
        p_same = (probs_t1 * probs_t2).sum(dim=1)  # (B, H, W)
        p_change = 1.0 - p_same

        target = change_gt
        p_t = p_change * target + (1 - p_change) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        bce = F.binary_cross_entropy(p_change.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        focal = (alpha_t * (1 - p_t).pow(self.gamma) * bce * valid).sum() / valid.sum().clamp_min(1)

        return focal


class SCDLoss(nn.Module):
    """Loss SCD : CE sémantique + focal binaire auxiliaire pour le changement.

    ``L = CE_t1 + CE_t2 + lambda_bcd * focal_bcd``
    """

    def __init__(
        self,
        n_classes: int,
        lambda_bcd: float = 10.0,
        bcd_alpha: float = 0.75,
        bcd_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 0,
        **_kwargs,  # ignore d'anciens paramètres (lambda_dice, etc.)
    ):
        super().__init__()
        self.n_classes = n_classes
        self.lambda_bcd = lambda_bcd
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index, reduction="none",
        )
        self.bcd = BinaryChangeLoss(alpha=bcd_alpha, gamma=bcd_gamma)

    def forward(
        self,
        logits: torch.Tensor,
        sem_t1: torch.Tensor,
        sem_t2: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        K = self.n_classes
        logits_t1 = logits[:, :K]
        logits_t2 = logits[:, K:]

        n_valid = valid.sum().clamp_min(1)
        ce_t1 = (self.ce(logits_t1, sem_t1) * valid).sum() / n_valid
        ce_t2 = (self.ce(logits_t2, sem_t2) * valid).sum() / n_valid

        # BCD auxiliaire : change_gt = pixels où les classes GT diffèrent
        sem_valid = valid * (sem_t1 != self.ignore_index).float() * (sem_t2 != self.ignore_index).float()
        change_gt = (sem_t1 != sem_t2).float()

        probs_t1 = torch.softmax(logits_t1, dim=1)
        probs_t2 = torch.softmax(logits_t2, dim=1)
        bcd_loss = self.bcd(probs_t1, probs_t2, change_gt, sem_valid)

        return (ce_t1 + ce_t2) + self.lambda_bcd * bcd_loss
