from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

from .fcn import ResNet18FCN
from .man import MultiHeadAttentionNet
from .afn import AttentionFusionNet
from .losses import JointLoss, LossWeights


@dataclass
class MHCAConfig:
    in_channels: int = 3
    num_classes: int = 7
    num_heads: int = 4
    embed_dim: int = 256
    temperature: float = 1.0
    score_from_vec: bool = False
    pool: str = "gap"  # gap/gmp/avgmax
    pretrained_backbone: bool = True
    # loss weights
    w_cls: float = 1.0
    w_affinity: float = 0.1
    w_partition: float = 0.1
    w_part_conf: float = 1.0
    w_part_balance: float = 1.0

class MHCAEmotionNet(nn.Module):
    """
    Multi-Head Cross Attention (approximation) for Facial Expression Recognition.

    Components:
      - FCN: ResNet-18 backbone producing spatial feature maps (B, 512, H/32, W/32).
      - MAN: Parallel attention heads; each produces an embedding and per-class logits.
      - AFN: Fuses head logits using log-softmax over head scores; provides weights for partition loss.

    Losses:
      - Classification: Cross Entropy on fused logits.
      - Affinity: Spatial smoothness on FCN feature map.
      - Partition: Entropy-based regulation of head weights.

    Forward signatures:
      - model(x) -> dict(logits, aux)
      - model(x, targets) -> dict(logits, loss, losses, aux)

    Jupyter-friendly example:
      >>> import torch
      >>> from model.mhca import MHCAEmotionNet, MHCAConfig
      >>> cfg = MHCAConfig(num_classes=7, num_heads=4)
      >>> model = MHCAEmotionNet(cfg)
      >>> x = torch.randn(2, 3, 224, 224)
      >>> out = model(x)
      >>> out['logits'].shape  # (2, 7)
    """

    def __init__(self, config: MHCAConfig):
        super().__init__()
        self.config = config

        # FCN backbone
        self.fcn = ResNet18FCN(in_ch=config.in_channels, pretrained=config.pretrained_backbone)

        # MAN: heads operate on 512-channel feature map from ResNet18 layer4
        self.man = MultiHeadAttentionNet(
            in_channels=512,
            num_classes=config.num_classes,
            num_heads=config.num_heads,
            embed_dim=config.embed_dim,
            pool=config.pool,
        )

        # AFN: fuse head logits
        self.afn = AttentionFusionNet(
            num_heads=config.num_heads,
            embed_dim=config.embed_dim,
            temperature=config.temperature,
            score_from_vec=config.score_from_vec,
        )

        # Loss combiner
        lw = LossWeights(
            cls=config.w_cls,
            affinity=config.w_affinity,
            partition=config.w_partition,
            part_conf=config.w_part_conf,
            part_balance=config.w_part_balance,
        )
        self.joint_loss = JointLoss(lw)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        aux: Dict[str, Any] = {}

        # 1) FCN feature map
        feat = self.fcn(x)  # (B,512,H/32,W/32)
        aux["fcn_feature"] = feat

        # 2) MAN heads
        head_logits, head_vecs, head_scores, attn_aux = self.man(feat)
        aux.update(attn_aux)
        aux["head_logits"] = head_logits
        aux["head_scores_raw"] = head_scores
        aux["head_vecs"] = head_vecs

        # 3) AFN fusion
        fused_logits, head_log_weights, afn_aux = self.afn(head_logits, head_scores, head_vecs)
        aux["head_weights"] = afn_aux["weights"]
        aux["head_log_weights"] = head_log_weights

        out: Dict[str, Any] = {"logits": fused_logits, "aux": aux}

        # 4) Losses (optional)
        if targets is not None:
            total, loss_dict = self.joint_loss(
                logits=fused_logits,
                targets=targets,
                feat=feat,
                head_log_weights=head_log_weights,
                temperature=self.config.temperature,
            )
            out["loss"] = total
            out["losses"] = loss_dict
        return out

    def infer_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return softmax probabilities from logits."""
        return torch.softmax(self.forward(x)["logits"], dim=-1)

    def freeze_backbone(self, freeze: bool = True):
        for p in self.fcn.parameters():
            p.requires_grad = not freeze

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
