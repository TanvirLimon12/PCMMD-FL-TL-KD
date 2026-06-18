"""
transforms.py
-------------
ImageNet-style 224x224 resizing + tensor conversion + clinical stain normalisation.

Wright-Giemsa stain stats (PCMMD-specific, computed from EDA cell crops):
  mean = [0.6890, 0.5280, 0.7000]
  std  = [0.1700, 0.1900, 0.1500]

Falls back to ImageNet stats when use_pcmmd_stats=False (default for TL warm-up).
"""

from torchvision import transforms

# ── Normalisation constants ──────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# PCMMD Wright-Giemsa stats computed from EDA over Set-1 & Set-3 cell crops
PCMMD_MEAN = [0.6890, 0.5280, 0.7000]
PCMMD_STD  = [0.1700, 0.1900, 0.1500]

IMAGE_SIZE = 224  # fixed globally — DO NOT change

def get_train_transforms(use_pcmmd_stats: bool = False) -> transforms.Compose:
    """
    Augmentation policy:
      - Mild spatial: H-flip, V-flip, ±10° rotation.
      - Mild colour: brightness/contrast jitter ±10% — preserves stain hue.
      - NO hue/saturation jitter (destroys Wright-Giemsa discriminative colour).
    """
    mean = PCMMD_MEAN if use_pcmmd_stats else IMAGENET_MEAN
    std  = PCMMD_STD  if use_pcmmd_stats else IMAGENET_STD

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_val_transforms(use_pcmmd_stats: bool = False) -> transforms.Compose:
    """Deterministic pipeline for validation / test / FL evaluation."""
    mean = PCMMD_MEAN if use_pcmmd_stats else IMAGENET_MEAN
    std  = PCMMD_STD  if use_pcmmd_stats else IMAGENET_STD

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_tta_transforms(use_pcmmd_stats: bool = False, n_augments: int = 5):
    """
    Test-Time Augmentation: returns a list of `n_augments` deterministic
    transform pipelines with different flip/rotation combos for TTA averaging.
    """
    mean = PCMMD_MEAN if use_pcmmd_stats else IMAGENET_MEAN
    std  = PCMMD_STD  if use_pcmmd_stats else IMAGENET_STD

    variants = [
        [],
        [transforms.RandomHorizontalFlip(p=1.0)],
        [transforms.RandomVerticalFlip(p=1.0)],
        [transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)],
        [transforms.RandomRotation(degrees=(90, 90))],
    ]
    pipelines = []
    for extra in variants[:n_augments]:
        pipelines.append(transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))] + extra +
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        ))
    return pipelines