"""Pertes pour le Semantic Change Detection (SCD).

Le modèle prédit ``(B, 2K, H, W)`` : les K premiers canaux = carte
sémantique à T1, les K suivants = carte sémantique à T2.

On combine :
  1. CE + soft dice multi-classe (symétriques sur T1 et T2)
  2. Un terme auxiliaire BCD qui force le modèle à détecter les zones
     de changement (là où sem_t1 ≠ sem_t2). Sans ce terme, le gradient
     des ~1% de pixels changed est noyé par les ~99% unchanged et le
     modèle apprend le raccourci "même carte partout".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    """Soft dice multi-classe, moyenné sur les classes (hors ignore_index)."""

    def __init__(self, n_classes: int, ignore_index: int = 0, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        K = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        valid_4d = valid.unsqueeze(1)
        one_hot = F.one_hot(targets, num_classes=K).permute(0, 3, 1, 2).float()

        dice_sum = 0.0
        count = 0
        for c in range(K):
            if c == self.ignore_index:
                continue
            p = probs[:, c:c+1] * valid_4d
            t = one_hot[:, c:c+1] * valid_4d
            inter = (p * t).sum()
            denom = p.sum() + t.sum()
            dice_sum += (2 * inter + self.smooth) / (denom + self.smooth)
            count += 1
        return 1.0 - dice_sum / max(count, 1)


class BinaryChangeLoss(nn.Module):
    """Focal + Dice binaire sur le signal de changement dérivé des softmax.

    La probabilité de changement est calculée de manière différentiable :
    ``p_change = 1 - sum_c(softmax_t1_c * softmax_t2_c)`` — c'est 0 quand
    les distributions sont identiques et ~1 quand elles divergent.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self,
        probs_t1: torch.Tensor,
        probs_t2: torch.Tensor,
        change_gt: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """probs_t1/t2: (B, K, H, W) softmax, change_gt: (B, H, W) float {0,1}, valid: (B, H, W) float."""
        # Probabilité de changement différentiable
        p_same = (probs_t1 * probs_t2).sum(dim=1)  # (B, H, W)
        p_change = 1.0 - p_same

        # Focal loss binaire
        target = change_gt
        p_t = p_change * target + (1 - p_change) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        bce = F.binary_cross_entropy(p_change.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        focal = (alpha_t * (1 - p_t).pow(self.gamma) * bce * valid).sum() / valid.sum().clamp_min(1)

        # Dice binaire
        p_v = p_change * valid
        t_v = target * valid
        inter = (p_v * t_v).sum()
        denom = p_v.sum() + t_v.sum()
        dice = 1.0 - (2 * inter + self.smooth) / (denom + self.smooth)

        return focal + dice


class SCDLoss(nn.Module):
    """Loss SCD complète : sémantique (CE + dice) + auxiliaire BCD.

    ``L = L_sem_t1 + L_sem_t2 + lambda_dice * (dice_t1 + dice_t2) + lambda_bcd * L_bcd``
    """

    def __init__(
        self,
        n_classes: int,
        lambda_dice: float = 1.0,
        lambda_bcd: float = 10.0,
        bcd_alpha: float = 0.75,
        bcd_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.lambda_dice = lambda_dice
        self.lambda_bcd = lambda_bcd
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index, reduction="none",
        )
        self.dice = MultiClassDiceLoss(
            n_classes=n_classes, ignore_index=ignore_index,
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

        dice_t1 = self.dice(logits_t1, sem_t1, valid)
        dice_t2 = self.dice(logits_t2, sem_t2, valid)

        # BCD auxiliaire : change_gt = pixels où les classes GT diffèrent
        # On masque aussi les pixels où l'une des deux dates est ignore.
        sem_valid = valid * (sem_t1 != self.ignore_index).float() * (sem_t2 != self.ignore_index).float()
        change_gt = (sem_t1 != sem_t2).float()

        probs_t1 = torch.softmax(logits_t1, dim=1)
        probs_t2 = torch.softmax(logits_t2, dim=1)
        bcd_loss = self.bcd(probs_t1, probs_t2, change_gt, sem_valid)

        return (ce_t1 + ce_t2) + self.lambda_dice * (dice_t1 + dice_t2) + self.lambda_bcd * bcd_loss
