"""
generate_tsne.py
-----------------
t-SNE / UMAP feature-space visualization.

Extracts penultimate-layer features from a set of checkpoints, then plots
t-SNE (and optionally UMAP) coloured by (a) class label and (b) patient/client.

Compares:
  • Best centralized checkpoint (e.g. EfficientNet-B0)
  • FedAvg global checkpoint
  • FedBN global checkpoint
  • KD student checkpoint

Outputs (figures/tsne/):
  tsne_<tag>_by_class.png
  tsne_<tag>_by_patient.png
  tsne_all_methods_f1_overlay.png   — side-by-side 2×2 panel

Run:
  python generate_tsne.py --fold 1
  python generate_tsne.py --fold 1 --use_umap   # requires umap-learn
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import PCMMDDataset
from data.transforms import get_val_transforms
from models import build_model
from utils import get_device, load_config, set_seed

CLASS_COLORS  = {0: "#C44E52", 1: "#4C72B0"}  # plasma=0, non_plasma=1
CLASS_LABELS  = {0: "Plasma", 1: "Non-plasma"}
DIAG_COLORS   = {"mm": "#C44E52", "normal": "#55A868"}


# ── Feature extractor ─────────────────────────────────────────────────────────

class _Penultimate(nn.Module):
    """Wrap a model, return penultimate features (before final classifier)."""
    def __init__(self, model, backbone_name: str):
        super().__init__()
        self._model = model
        self._name  = backbone_name
        self._feats = None
        self._register_hook()

    def _register_hook(self):
        if self._name == "resnet50":
            layer = self._model.model.avgpool
        elif self._name == "efficientnet_b0":
            layer = self._model.model.avgpool
        elif self._name == "mobilenet_v3":
            layer = self._model.model.avgpool
        else:
            layer = list(self._model.modules())[-3]
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._feats = out.detach().flatten(1)

    def forward(self, x):
        _ = self._model(x)
        return self._feats


@torch.no_grad()
def extract_features(model, loader, device) -> tuple[np.ndarray, np.ndarray, list]:
    model.eval()
    feats, labels, pids = [], [], []
    for batch in loader:
        imgs = batch[0].to(device)
        lbls = batch[1].numpy()
        patient_ids = batch[2] if len(batch) > 2 else ["?"] * len(lbls)
        out = model(imgs)
        feats.append(out.cpu().numpy())
        labels.extend(lbls.tolist())
        pids.extend(patient_ids if isinstance(patient_ids, list)
                    else patient_ids.tolist())
    return np.concatenate(feats), np.array(labels), pids


def _reduce(feats: np.ndarray, method: str = "tsne") -> np.ndarray:
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(feats)
        except ImportError:
            print("[warn] umap-learn not installed, falling back to t-SNE")
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats) // 4)).fit_transform(feats)


def _scatter(ax, embed, colors, title, legend_patches=None):
    ax.scatter(embed[:, 0], embed[:, 1], c=colors, s=8, alpha=0.7, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=7, loc="upper right",
                  framealpha=0.6, markerscale=2)


def _plot_single(embed, labels, patient_ids, tag, out_dir: Path, method_str: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # by class
    colors_cls = [CLASS_COLORS[int(l)] for l in labels]
    patches_cls = [mpatches.Patch(color=CLASS_COLORS[k], label=v) for k, v in CLASS_LABELS.items()]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    _scatter(ax, embed, colors_cls, f"{method_str} — by class", patches_cls)
    fig.tight_layout(); fig.savefig(out_dir / f"tsne_{tag}_by_class.png", dpi=150)
    plt.close(fig)

    # by patient
    unique_pids = sorted(set(str(p) for p in patient_ids))
    pid_cmap = plt.colormaps["tab10"].resampled(len(unique_pids))
    pid_color_map = {p: pid_cmap(i) for i, p in enumerate(unique_pids)}
    colors_pid = [pid_color_map[str(p)] for p in patient_ids]
    patches_pid = [mpatches.Patch(color=pid_color_map[p], label=f"P{p}") for p in unique_pids]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    _scatter(ax, embed, colors_pid, f"{method_str} — by patient", patches_pid)
    fig.tight_layout(); fig.savefig(out_dir / f"tsne_{tag}_by_patient.png", dpi=150)
    plt.close(fig)

    print(f"  saved tsne_{tag}_by_class.png + tsne_{tag}_by_patient.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold",    type=int, default=1)
    ap.add_argument("--config",  default="configs/centralized.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--use_umap", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed()
    device = get_device()
    cfg    = load_config(args.config)
    fold   = args.fold
    out_dir = Path(args.figures) / "tsne"
    method  = "umap" if args.use_umap else "tsne"

    # ── Build test dataset ────────────────────────────────────────────────────
    fold_csv = Path(cfg["fold_dir"]) / f"fold_{fold}.csv" if cfg.get("fold_dir") else None
    if fold_csv is None or not fold_csv.exists():
        fold_csv = Path("data/eda") / f"fold_{fold}.csv"
    if not fold_csv.exists():
        print(f"[error] fold CSV not found: {fold_csv}")
        return

    dataset = PCMMDDataset(fold_csv, split="test",
                           image_root=cfg.get("image_root", "data/patient_cells"),
                           root_dir=cfg.get("root_dir", "./"),
                           transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # ── Checkpoints to compare ────────────────────────────────────────────────
    ckpt_root = Path(cfg.get("ckpt_dir", "checkpoints"))
    configs_to_run = [
        ("centralized", "efficientnet_b0",
         ckpt_root / "centralized" / f"efficientnet_b0_fold{fold}.pth",
         "EfficientNet (Centralized)"),
        ("centralized", "resnet50",
         ckpt_root / "centralized" / f"resnet50_fold{fold}.pth",
         "ResNet50 (Centralized)"),
        ("centralized", "mobilenet_v3",
         ckpt_root / "centralized" / f"mobilenet_v3_fold{fold}.pth",
         "MobileNetV3 (Centralized)"),
        ("fedavg", "mobilenet_v3",
         ckpt_root / "fedavg" / f"mobilenet_v3_noniid_fold{fold}.pth",
         "MobileNetV3 (FedAvg)"),
        ("fedbn", "mobilenet_v3",
         ckpt_root / "fedbn" / f"mobilenet_v3_noniid_fold{fold}.pth",
         "MobileNetV3 (FedBN)"),
        ("kd", "mobilenet_v3",
         ckpt_root / "kd" / f"mobilenet_v3_distilled_fold{fold}.pth",
         "MobileNetV3 (KD Student)"),
    ]

    all_embeds, all_labels_list, panel_titles, panel_tags = [], [], [], []

    for tag, backbone, ckpt_path, display in configs_to_run:
        if not ckpt_path.exists():
            print(f"  [skip] {ckpt_path} not found")
            continue
        print(f"  loading {display} ← {ckpt_path.name}")
        model = build_model(backbone, num_classes=2, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        extractor = _Penultimate(model, backbone).to(device)
        feats, labels_arr, pids = extract_features(extractor, loader, device)
        embed = _reduce(feats, method)
        _plot_single(embed, labels_arr, pids, f"{tag}_{backbone}", out_dir, display)
        all_embeds.append(embed)
        all_labels_list.append(labels_arr)
        panel_titles.append(display)
        panel_tags.append(f"{tag}_{backbone}")

    # ── 2×N panel all methods ─────────────────────────────────────────────────
    if len(all_embeds) >= 2:
        n = len(all_embeds)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        patches_cls = [mpatches.Patch(color=CLASS_COLORS[k], label=v) for k, v in CLASS_LABELS.items()]
        for ax, embed, labels_arr, title in zip(axes, all_embeds, all_labels_list, panel_titles):
            colors_cls = [CLASS_COLORS[int(l)] for l in labels_arr]
            _scatter(ax, embed, colors_cls, title, patches_cls)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(f"{method.upper()} Feature Space Comparison — Fold {fold}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "tsne_all_methods_panel.png", dpi=150)
        plt.close(fig)
        print(f"  saved → {out_dir}/tsne_all_methods_panel.png")

    if not all_embeds:
        print("\nNo checkpoints found. Run training first, then re-run this script.")
    else:
        print(f"\n{method.upper()} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
