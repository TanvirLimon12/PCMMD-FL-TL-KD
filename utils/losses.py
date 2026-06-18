"""
utils/losses.py
---------------
Loss factory for the centralized / few-shot tracks.

  ce          : plain CrossEntropy
  weighted_ce : class-weighted CrossEntropy (weights from train class frequencies)
  focal       : Focal loss (gamma=2 default) for hard-example emphasis under imbalance

Report which loss was used (Abrar workflow §A2 requires it) — the chosen name is
saved in the config snapshot of every run.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)                       # prob of the true class
        return ((1.0 - pt) ** self.gamma * ce).mean()


def build_loss(name: str = "ce", class_weights: Optional[torch.Tensor] = None,
               focal_gamma: float = 2.0) -> nn.Module:
    name = (name or "ce").lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name in ("weighted_ce", "weighted"):
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    raise ValueError(f"Unknown loss '{name}'. Use ce | weighted_ce | focal.")
