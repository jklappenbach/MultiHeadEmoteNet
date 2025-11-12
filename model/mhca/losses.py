from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    cls: float = 1.0
    affinity: float = 0.1
    partition: float = 0.1
    # partition sub-weights
    part_conf: float = 1.0  # encourage confident per-sample assignments (low entropy)
    part_balance: float = 1.0  # encourage balanced head utilization (high entropy across batch)


def affinity_loss(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Affinity loss over a feature map encouraging local spatial smoothness in embedding space.
    Implements 4-neighborhood cosine similarity maximization.

    Args:
        feat: (B, C, H, W) feature map.
        eps: numerical stability.
    Returns:
        Scalar loss (1 - mean cosine similarity to neighbors).
    """
    if feat.ndim != 4:
        raise ValueError("affinity_loss expects a 4D tensor (B,C,H,W)")
    # normalize features per-location
    x = F.normalize(feat, p=2, dim=1)  # (B,C,H,W)

    # Shifted versions for 4-neighborhood
    def _pad_shift(t, pad, shift_slices):
        t = F.pad(t, pad, mode="replicate")
        return t[shift_slices]

    # Right neighbor (shift left)
    xr = _pad_shift(x, pad=(1, 0, 0, 0), shift_slices=(slice(None), slice(None), slice(None), slice(0, -1)))
    # Left neighbor (shift right)
    xl = _pad_shift(x, pad=(0, 1, 0, 0), shift_slices=(slice(None), slice(None), slice(None), slice(1, None)))
    # Down neighbor (shift up)
    xd = _pad_shift(x, pad=(0, 0, 1, 0), shift_slices=(slice(None), slice(None), slice(0, -1), slice(None)))
    # Up neighbor (shift down)
    xu = _pad_shift(x, pad=(0, 0, 0, 1), shift_slices=(slice(None), slice(None), slice(1, None), slice(None)))

    sims = []
    for nbr in (xr, xl, xd, xu):
        # cosine similarity along C
        sim = (x * nbr).sum(dim=1)  # (B,H,W)
        sims.append(sim)
    sims = torch.stack(sims, dim=0)  # (4,B,H,W)
    mean_sim = sims.mean()
    loss = 1.0 - mean_sim.clamp(min=-1.0, max=1.0)
    return loss


def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp(min=eps)
    return -(p * (p.log())).sum(dim=-1)


def partition_loss(head_log_weights: torch.Tensor,
                   weights: Optional[LossWeights] = None,
                   temperature: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """
    Partition loss to regulate fusion weights across heads.

    Args:
        head_log_weights: (B, H) log-weights per head (e.g., from log_softmax over heads).
        weights: LossWeights container with part_conf and part_balance scalars.
        temperature: scaling that was applied before log_softmax; for accounting, not used here.

    Returns:
        (loss, stats_dict)
    """
    if weights is None:
        weights = LossWeights()

    # Convert to probabilities
    p = head_log_weights.exp()  # (B,H)

    # Per-sample entropy (we want it low for confident routing)
    ent_per_sample = _entropy(p)  # (B,)
    conf_term = ent_per_sample.mean()

    # Balance entropy across batch (we want it high to encourage utilization of all heads)
    mean_p = p.mean(dim=0)  # (H,)
    balance_ent = _entropy(mean_p.unsqueeze(0)).squeeze(0)  # scalar

    loss = weights.part_conf * conf_term - weights.part_balance * balance_ent

    stats = {
        "part_conf_entropy": conf_term.detach(),
        "part_balance_entropy": balance_ent.detach(),
        "part_temperature": torch.tensor(temperature, dtype=p.dtype, device=p.device),
    }
    return loss, stats


class JointLoss(nn.Module):
    """Combine classification CE, affinity loss, and partition loss."""

    def __init__(self, loss_weights: Optional[LossWeights] = None):
        super().__init__()
        self.weights = loss_weights or LossWeights()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                feat: Optional[torch.Tensor] = None,
                head_log_weights: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> Tuple[torch.Tensor, dict]:
        losses = {}

        cls = self.criterion(logits, targets)
        losses["loss_cls"] = cls * self.weights.cls

        if feat is not None:
            affin = affinity_loss(feat)
            losses["loss_affinity"] = affin * self.weights.affinity
        else:
            losses["loss_affinity"] = torch.tensor(0.0, device=logits.device)

        if head_log_weights is not None:
            part, part_stats = partition_loss(head_log_weights, self.weights, temperature)
            losses["loss_partition"] = part * self.weights.partition
        else:
            part_stats = {"part_conf_entropy": torch.tensor(0.0, device=logits.device),
                          "part_balance_entropy": torch.tensor(0.0, device=logits.device),
                          "part_temperature": torch.tensor(temperature, device=logits.device)}
            losses["loss_partition"] = torch.tensor(0.0, device=logits.device)

        total = sum(losses.values())
        losses["loss_total"] = total
        # attach stats
        losses.update(part_stats)
        return total, losses
