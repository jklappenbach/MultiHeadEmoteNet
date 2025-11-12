from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Simple channel attention similar to SE/CBAM style.
    Produces a channel-wise attention vector in [0,1].
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = (self.mlp(avg_pool) + self.mlp(max_pool)) * 0.5
        return attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial attention similar to CBAM: uses average and max along channels.
    Produces a spatial attention map in [0,1].
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        s = self.conv(s)
        s = self.bn(s)
        s = self.sigmoid(s)
        return s  # (B,1,H,W)


class AttentionHead(nn.Module):
    """
    One attention head combining channel and spatial attention paths.
    - Computes channel attention then spatial attention.
    - Returns attended feature map and a pooled feature vector.
    """

    def __init__(self, channels: int, embed_dim: int, spatial_kernel: int = 7, pool: str = "gap"):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)
        self.pool = pool
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        ca = self.ca(x)  # (B,C,1,1)
        xs = x * ca
        sa = self.sa(xs)  # (B,1,H,W)
        xsa = xs * sa
        y = self.act(self.bn(self.proj(xsa)))  # (B, E, H, W)
        if self.pool == "gap":
            vec = F.adaptive_avg_pool2d(y, 1).flatten(1)
        elif self.pool == "gmp":
            vec = F.adaptive_max_pool2d(y, 1).flatten(1)
        else:
            vec = (F.adaptive_avg_pool2d(y, 1) + F.adaptive_max_pool2d(y, 1)).flatten(1) * 0.5
        return y, vec, ca.squeeze(-1).squeeze(-1), sa.squeeze(1)
