"""Training losses for the row-anchor formulation.

* Classification loss (the main one): for every slot and row anchor the
  target is one of C+1 classes, so we use ordinary cross-entropy. Note it
  treats "bin 41 instead of 42" as just as wrong as "bin 3 instead of 42";
  the smoothness term and the sub-bin decoding soften that.

* Structural smoothness loss (optional, UFLD's "similarity loss"): real lane
  lines are continuous, so the predicted distribution over columns on one row
  should look like the one on the next row. We penalise the L1 distance
  between the softmax distributions of adjacent rows, only where the ground
  truth says the lane is present on both rows (so lane ends are not blurred).

* Exists loss (optional): binary cross-entropy on "does this slot contain a
  lane at all?".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def row_classification_loss(row_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """row_logits (B, S, R, C+1), target (B, S, R) int64 -> scalar mean cross-entropy."""
    k = row_logits.shape[-1]
    return F.cross_entropy(row_logits.reshape(-1, k), target.reshape(-1))


def smoothness_loss(row_logits: torch.Tensor, target: torch.Tensor, absent_class: int) -> torch.Tensor:
    """Mean L1 distance between column distributions of adjacent present rows."""
    p = F.softmax(row_logits[..., :absent_class], dim=-1)
    diff = (p[:, :, 1:] - p[:, :, :-1]).abs().sum(dim=-1)  # (B, S, R-1)
    present = target != absent_class
    both = (present[:, :, 1:] & present[:, :, :-1]).float()
    return (diff * both).sum() / both.sum().clamp(min=1.0)


def exists_loss(exists_logits: torch.Tensor, exists_target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(exists_logits, exists_target)


class LaneLoss:
    """Weighted sum of the three terms; returns (total, {name: value}) for logging."""

    def __init__(self, absent_class: int, smooth_weight: float = 0.5, exists_weight: float = 0.1):
        self.absent_class = absent_class
        self.smooth_weight = smooth_weight
        self.exists_weight = exists_weight

    def __call__(
        self,
        row_logits: torch.Tensor,
        exists_logits: torch.Tensor,
        cls_target: torch.Tensor,
        exists_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cls = row_classification_loss(row_logits, cls_target)
        total = cls
        parts = {"cls": float(cls.detach())}
        if self.smooth_weight > 0:
            sm = smoothness_loss(row_logits, cls_target, self.absent_class)
            total = total + self.smooth_weight * sm
            parts["smooth"] = float(sm.detach())
        if self.exists_weight > 0:
            ex = exists_loss(exists_logits, exists_target)
            total = total + self.exists_weight * ex
            parts["exists"] = float(ex.detach())
        parts["total"] = float(total.detach())
        return total, parts
