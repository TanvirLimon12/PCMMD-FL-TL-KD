"""
explainability.py
-----------------
Grad-CAM visualisations for PCMMD models (ResNet50, EfficientNet-B0, MobileNetV3).

Generates heatmap overlays on representative test-set images:
  - 2 true positives  (plasma correctly classified)
  - 2 true negatives  (non-plasma correctly classified)
  - 2 false negatives (plasma missed — highest-priority errors clinically)
  - 2 false positives (non-plasma misclassified as plasma)

Saves:
  figures/explainability/<tag>_gradcam_panel.png
  figures/explainability/<tag>_gradcam_<category>_<n>.png  (individual)

Usage:
  python explainability.py --config configs/centralized.yaml \
      --weights checkpoints/centralized/efficientnet_b0_fold1.pth \
      --backbone efficientnet_b0 --fold 1 --tag effnet

  python explainability.py --config configs/kd.yaml \
      --weights checkpoints/kd/distilled_T4.0_mobilenet_v3_fold1.pth \
      --backbone mobilenet_v3 --fold 1 --tag student_kd
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.folds import get_fold_loaders  # noqa: E402
from data.dataset import CLASS_TO_IDX  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_config, set_seed  # noqa: E402

POSITIVE_CLASS_IDX = 0   # plasma


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

def _get_target_layer(model: torch.nn.Module, backbone: str) -> torch.nn.Module:
    """Return the last conv/feature layer for gradient hooks."""
    inner = model.model   # unwrap EfficientNetB0 / MobileNetV3 / ResNet50 wrapper
    if backbone == "resnet50":
        return inner.layer4[-1]
    elif backbone == "efficientnet_b0":
        return inner.features[-1]
    elif backbone == "mobilenet_v3":
        return inner.features[-1]
    else:
        raise ValueError(f"Unknown backbone for Grad-CAM: {backbone}")


class GradCAM:
    """Hook-based Grad-CAM for a single target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self._feats: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._fh = target_layer.register_forward_hook(self._save_feats)
        self._bh = target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, module, inp, out):
        self._feats = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def __call__(self, img_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        img_tensor : (1, C, H, W) on device
        Returns    : (H, W) heatmap in [0, 1]
        """
        self.model.eval()
        img_tensor = img_tensor.requires_grad_(True)
        logits = self.model(img_tensor)
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Global average pooling of gradients over spatial dims
        weights = self._grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._feats).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.ndim == 0:
            cam = np.expand_dims(cam, 0)
        # Resize to 224×224 via bilinear
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_r = F.interpolate(cam_t, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam_r.squeeze().numpy()
        vmin, vmax = cam_np.min(), cam_np.max()
        if vmax > vmin:
            cam_np = (cam_np - vmin) / (vmax - vmin)
        return cam_np.astype(np.float32)


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_rgb(path: str | Path, size: int = 224) -> np.ndarray:
    """Load image as (H, W, 3) uint8 RGB, resized."""
    try:
        img = Image.open(path).convert("RGB").resize((size, size))
        return np.array(img)
    except Exception:
        return np.full((size, size, 3), 200, dtype=np.uint8)


# ── Overlay ───────────────────────────────────────────────────────────────────

def _overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend Grad-CAM heatmap (jet) with the original RGB image."""
    cmap = plt.get_cmap("jet")
    heat = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * heat + (1 - alpha) * rgb).clip(0, 255).astype(np.uint8)
    return blended


# ── Prediction pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def _score_loader(model, loader, device):
    """Return list of (image_path, true_label, pred_label, prob_plasma, img_tensor)."""
    model.eval()
    records = []
    for batch in loader:
        imgs, lbls = batch[0], batch[1]
        paths = list(batch[3]) if len(batch) > 3 else [""] * len(lbls)
        imgs_dev = imgs.to(device)
        logits = model(imgs_dev)
        probs = F.softmax(logits, dim=1)[:, POSITIVE_CLASS_IDX]
        preds = logits.argmax(dim=1)
        for i in range(len(lbls)):
            records.append({
                "path": paths[i],
                "true": int(lbls[i]),
                "pred": int(preds[i].cpu()),
                "prob": float(probs[i].cpu()),
                "img_tensor": imgs[i].unsqueeze(0),   # CPU (1,C,H,W)
            })
    return records


def _pick(records, condition, n: int = 2, sort_key=None, reverse=True):
    subset = [r for r in records if condition(r)]
    if sort_key:
        subset = sorted(subset, key=sort_key, reverse=reverse)
    return subset[:n]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/centralized.yaml")
    ap.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--backbone", default="efficientnet_b0",
                    choices=["resnet50", "efficientnet_b0", "mobilenet_v3"])
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--tag", default="gradcam")
    ap.add_argument("--n_per_cat", type=int, default=2,
                    help="Number of examples per category (TP/TN/FP/FN)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed()
    device = get_device()

    model = build_model(args.backbone, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    _, _, test_loader = get_fold_loaders(
        fold_dir=cfg["fold_dir"], fold_id=args.fold,
        batch_size=cfg.get("batch_size", 32), num_workers=cfg.get("num_workers", 2),
        root_dir=cfg.get("root_dir"), image_root=cfg.get("image_root"),
        use_weighted_sampler=False)

    records = _score_loader(model, test_loader, device)

    categories = {
        "TP": _pick(records, lambda r: r["true"] == 0 and r["pred"] == 0,
                    args.n_per_cat, sort_key=lambda r: r["prob"]),
        "TN": _pick(records, lambda r: r["true"] == 1 and r["pred"] == 1,
                    args.n_per_cat, sort_key=lambda r: r["prob"], reverse=False),
        "FN": _pick(records, lambda r: r["true"] == 0 and r["pred"] == 1,
                    args.n_per_cat, sort_key=lambda r: r["prob"], reverse=False),
        "FP": _pick(records, lambda r: r["true"] == 1 and r["pred"] == 0,
                    args.n_per_cat, sort_key=lambda r: r["prob"]),
    }

    target_layer = _get_target_layer(model, args.backbone)
    gcam = GradCAM(model, target_layer)

    out_dir = Path(cfg.get("figures_dir", "figures")) / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_images: List[Tuple[str, np.ndarray]] = []
    label_names = {0: "plasma", 1: "non_plasma"}

    for cat, recs in categories.items():
        for i, rec in enumerate(recs):
            img_t = rec["img_tensor"].to(device)
            # Target class = predicted class for CAM (shows what drives the prediction)
            cam = gcam(img_t, target_class=rec["pred"])
            rgb = _load_rgb(rec["path"]) if rec["path"] else np.full((224, 224, 3), 200, dtype=np.uint8)
            overlay = _overlay(rgb, cam)
            title = (f"{cat} | true={label_names[rec['true']]} "
                     f"pred={label_names[rec['pred']]} p={rec['prob']:.2f}")
            all_images.append((title, overlay))

            # Save individual
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(rgb); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(cam, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
            axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
            fig.suptitle(title, fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / f"{args.tag}_{cat}_{i}.png", dpi=150)
            plt.close(fig)
            print(f"Saved {cat}_{i}: {title}")

    gcam.remove()

    # Panel: all categories in a grid
    n_rows = len(categories)
    n_cols = args.n_per_cat
    if all_images:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
        if n_rows == 1:
            axes = [axes]
        for ax_row, (cat, recs) in zip(axes, categories.items()):
            if not isinstance(ax_row, (list, np.ndarray)):
                ax_row = [ax_row]
            for j, (ax, rec) in enumerate(zip(ax_row, recs)):
                img_idx = list(categories.keys()).index(cat) * args.n_per_cat + j
                if img_idx < len(all_images):
                    title, overlay = all_images[img_idx]
                    ax.imshow(overlay)
                    ax.set_title(title, fontsize=7)
                ax.axis("off")
        fig.suptitle(f"Grad-CAM Panel — {args.backbone} fold {args.fold}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.tag}_gradcam_panel.png", dpi=150)
        plt.close(fig)
        print(f"\nPanel saved: {out_dir}/{args.tag}_gradcam_panel.png")


if __name__ == "__main__":
    main()
