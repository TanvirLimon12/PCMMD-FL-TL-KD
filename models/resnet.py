"""ResNet-50 wrapper for PCMMD binary classification."""
import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = get_resnet50(num_classes, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)