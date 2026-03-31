import torch.nn as nn
import torch.nn.functional as F


class BaselineCLS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t1, t2):
        cls1 = F.normalize(t1[:, 0, :], dim=-1)
        cls2 = F.normalize(t2[:, 0, :], dim=-1)
        return (cls1 * cls2).sum(dim=-1)

    @property
    def trainable(self):
        return False


class BaselinePatches(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t1, t2):
        patches1 = F.normalize(t1[:, 1:, :], dim=-1)
        patches2 = F.normalize(t2[:, 1:, :], dim=-1)
        return (patches1 * patches2).sum(dim=-1).mean(dim=-1)

    @property
    def trainable(self):
        return False
