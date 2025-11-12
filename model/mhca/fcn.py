from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet18
except Exception:  # pragma: no cover - allow usage without torchvision at import time
    resnet18 = None  # type: ignore


class ResNet18FCN(nn.Module):
    """
    Feature Clustering Network (FCN):
    - ResNet-18 backbone truncated to produce a spatial feature map from layer4.
    - Optionally adapts input channels via a 1x1 conv if in_ch != 3.
    - Exposes the feature map for downstream attention heads.

    Output shape: (B, 512, H/32, W/32)
    """

    def __init__(self, in_ch: int = 3, pretrained: bool = True):
        super().__init__()
        if resnet18 is None:
            raise ImportError("torchvision is required to use ResNet18FCN. Please install torchvision.")
        base = resnet18(weights=None if not pretrained else None)  # keep simple; user can load weights later
        # Note: For torchvision>=0.13, use weights arg; keeping None to avoid hard dependency on version.
        # Adapt the first conv if needed.
        if in_ch != 3:
            old_conv = base.conv1
            self.input_adapter = nn.Conv2d(in_ch, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                           stride=old_conv.stride, padding=old_conv.padding, bias=False)
        else:
            self.input_adapter = None

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        ) if self.input_adapter is None else nn.Sequential(
            self.input_adapter,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        # remove avgpool and fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B,512,H/32,W/32)
