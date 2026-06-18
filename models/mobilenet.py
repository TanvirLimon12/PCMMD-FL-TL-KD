"""MobileNetV3-Small wrapper for PCMMD binary classification."""
import torch
import torch.nn as nn
from torchvision import models

def get_mobilenet_v3(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

class MobileNetV3(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = get_mobilenet_v3(num_classes, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)