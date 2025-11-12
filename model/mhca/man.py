from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AttentionHead


class MultiHeadAttentionNet(nn.Module):
    """
    Multi-head Attention Network (MAN)
    - Spawns H parallel attention heads operating on the same feature map.
    - Each head returns an embedding vector and per-class logits.
    - Additionally, computes a per-head scalar score used by AFN for fusion.

    Returns from forward:
        head_logits: (B, H, C)
        head_vecs: (B, H, E)
        head_scores: (B, H)  # higher means stronger head
        aux_attn: dict with optional attention maps per head
    """

    def __init__(self, in_channels: int, num_classes: int, num_heads: int = 4, embed_dim: int = 256,
                 pool: str = "gap"):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.heads = nn.ModuleList([
            AttentionHead(in_channels, embed_dim, pool=pool) for _ in range(num_heads)
        ])
        # classifier per head
        self.classifiers = nn.ModuleList([
            nn.Linear(embed_dim, num_classes) for _ in range(num_heads)
        ])
        # head scorers (a simple linear on head vector)
        self.scorers = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        head_vecs: List[torch.Tensor] = []
        head_logits: List[torch.Tensor] = []
        head_scores: List[torch.Tensor] = []
        ca_list: List[torch.Tensor] = []
        sa_list: List[torch.Tensor] = []

        for h, clf, scr in zip(self.heads, self.classifiers, self.scorers):
            y, vec, ca, sa = h(x)
            head_vecs.append(vec)
            head_logits.append(clf(vec))
            head_scores.append(scr(vec).squeeze(-1))
            ca_list.append(ca)
            sa_list.append(sa)

        head_vecs_t = torch.stack(head_vecs, dim=1)  # (B,H,E)
        head_logits_t = torch.stack(head_logits, dim=1)  # (B,H,C)
        head_scores_t = torch.stack(head_scores, dim=1)  # (B,H)
        aux = {
            "channel_attn": torch.stack(ca_list, dim=1),  # (B,H,C)
            "spatial_attn": torch.stack(sa_list, dim=1),  # (B,H,Hs,Ws)
        }
        return head_logits_t, head_vecs_t, head_scores_t, aux
