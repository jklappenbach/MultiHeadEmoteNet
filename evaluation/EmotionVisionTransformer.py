#%% md
# Vision Transformers and Model Builders

#%%
import torchvision.models as tv_models
import torch.nn as nn

# Introduce a single source of truth for class count
NUM_CLASSES = 6

class EmotionViT(nn.Module):
    """
    Vision Transformer for emotion recognition built on torchvision's ViT-B/16.
    Replaces the classification head to output NUM_CLASSES logits.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, weights: tv_models.ViT_B_16_Weights | None = None):
        super().__init__()
        # Initialize base ViT (optionally with torchvision weights)
        base = tv_models.vit_b_16(weights=weights)
        # Replace classifier head
        in_features = base.heads.head.in_features
        base.heads.head = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

def build_emotion_vit_model(num_classes: int = NUM_CLASSES, *, use_pretrained: bool = False) -> nn.Module:
    weights = tv_models.ViT_B_16_Weights.DEFAULT if use_pretrained else None
    return EmotionViT(num_classes=num_classes, weights=weights)