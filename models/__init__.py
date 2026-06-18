from .efficientnet import EfficientNetB0, get_efficientnet_b0
from .mobilenet import MobileNetV3, get_mobilenet_v3
from .resnet import ResNet50, get_resnet50

MODEL_REGISTRY = {
    "efficientnet_b0": EfficientNetB0,
    "mobilenet_v3":    MobileNetV3,
    "resnet50":        ResNet50,
}

# Classifier-submodule name hints per backbone — keep the head trainable when the
# feature extractor is frozen (transfer-learning protocol, proposal §7.2 / E3).
_HEAD_HINTS = ("fc", "classifier")


def apply_finetune_mode(model, mode: str = "full"):
    """
    full    : train everything (default)
    frozen  : freeze backbone, train classifier head only
    partial : freeze ~first 70% of parameters, fine-tune the rest + head
    """
    mode = (mode or "full").lower()
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    named = list(model.named_parameters())
    if mode == "frozen":
        for name, p in named:
            p.requires_grad_(any(h in name for h in _HEAD_HINTS))
    elif mode == "partial":
        cut = int(len(named) * 0.7)
        for i, (name, p) in enumerate(named):
            p.requires_grad_(i >= cut or any(h in name for h in _HEAD_HINTS))
    else:
        raise ValueError(f"Unknown finetune mode '{mode}'. Use full | frozen | partial.")
    return model


def build_model(name: str, num_classes: int = 2, pretrained: bool = True, finetune_mode: str = "full"):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)
    return apply_finetune_mode(model, finetune_mode)
