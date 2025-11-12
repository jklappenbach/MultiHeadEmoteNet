from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusionNet(nn.Module):
    """
    AFN: Fuse per-head logits using log-softmax-scaled scores.
    Optionally refine scores from head vectors via a small gating MLP.

    Inputs:
        head_logits: (B,H,C)
        head_scores: (B,H)  raw head scores
        head_vecs:   (B,H,E) optional, used when score_from_vec=True

    Returns:
        fused_logits: (B,C)
        head_log_weights: (B,H)  log-softmax over heads (for partition loss)
        aux: dict with 'weights' (B,H)
    """

    def __init__(self, num_heads: int, embed_dim: int, temperature: float = 1.0, score_from_vec: bool = False):
        super().__init__()
        self.temperature = temperature
        self.score_from_vec = score_from_vec
        if score_from_vec:
            self.gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2 if embed_dim > 1 else 1),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim // 2 if embed_dim > 1 else 1, 1),
            )
        else:
            self.gate = None
        self.num_heads = num_heads

    def forward(self,
                head_logits: torch.Tensor,
                head_scores: torch.Tensor,
                head_vecs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # head_logits: (B,H,C), head_scores: (B,H)
        if self.score_from_vec:
            assert head_vecs is not None, "head_vecs required when score_from_vec=True"
            # compute new scores per head
            B, H, E = head_vecs.shape
            s = self.gate(head_vecs.reshape(B * H, E)).reshape(B, H)
        else:
            s = head_scores

        log_w = F.log_softmax(s / max(1e-6, self.temperature), dim=1)  # (B,H)
        w = log_w.exp().unsqueeze(-1)  # (B,H,1)
        fused = (head_logits * w).sum(dim=1)  # (B,C)
        aux = {"weights": w.squeeze(-1)}
        return fused, log_w, aux
