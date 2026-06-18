"""EfficientNet-B0 wrapper for PCMMD binary classification."""
import torch
import torch.nn as nn
from torchvision import models

def get_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = get_efficientnet_b0(num_classes, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)