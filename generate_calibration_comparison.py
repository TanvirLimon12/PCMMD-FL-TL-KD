"""
generate_calibration_comparison.py
------------------------------------
Compare calibration (ECE) and confidence distributions across all methods.

Loads each available checkpoint, runs inference on fold TEST split, then:
  1. Computes ECE per method → results/calibration_comparison.csv
  2. Plots reliability diagrams side-by-side
  3. Plots confidence score distribution histograms per method
  4. Plots ECE bar chart across methods

Figures saved to figures/calibration/.

Run:
  python generate_calibration_comparison.py --fold 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import PCMMDDataset
from data.transforms import get_val_transforms
from models import build_model
from utils import expected_calibration_error, get_device, load_config, set_seed


@torch.no_grad()
def get_probs_and_labels(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        lbls = batch[1].numpy()
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)[:, 0].cpu().numpy()  # P(plasma)
        all_probs.extend(probs.tolist())
        all_labels.extend(lbls.tolist())
    return np.array(all_probs), np.array(all_labels)


def _reliability_bins(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"confidence": (lo + hi) / 2, "accuracy": np.nan, "count": 0})
        else:
            rows.append({"confidence": (lo + hi) / 2,
                         "accuracy": float((labels[mask] == 0).mean()),
                         "count": count})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold",    type=int, default=1)
    ap.add_argument("--config",  default="configs/centralized.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--n_bins",  type=int, default=10)
    args = ap.parse_args()

    set_seed()
    device = get_device()
    cfg    = load_config(args.config)
    fold   = args.fold
    out_dir = Path(args.figures) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_csv = Path(cfg.get("fold_dir") or "data/eda") / f"fold_{fold}.csv"
    if not fold_csv.exists():
        print(f"[error] {fold_csv} not found"); return

    dataset = PCMMDDataset(fold_csv, split="test",
                           image_root=cfg.get("image_root", "data/patient_cells"),
                           root_dir=cfg.get("root_dir", "./"),
                           transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=cfg.get("num_workers", 2))

    ckpt_root = Path(cfg.get("ckpt_dir", "checkpoints"))
    checkpoints = [
        ("EfficientNet\n(Centralized)", "efficientnet_b0",
         ckpt_root / "centralized" / f"efficientnet_b0_fold{fold}.pth"),
        ("ResNet50\n(Centralized)", "resnet50",
         ckpt_root / "centralized" / f"resnet50_fold{fold}.pth"),
        ("MobileNetV3\n(Centralized)", "mobilenet_v3",
         ckpt_root / "centralized" / f"mobilenet_v3_fold{fold}.pth"),
        ("MobileNetV3\n(FedAvg)", "mobilenet_v3",
         ckpt_root / "fedavg" / f"mobilenet_v3_noniid_fold{fold}.pth"),
        ("MobileNetV3\n(FedBN)", "mobilenet_v3",
         ckpt_root / "fedbn" / f"mobilenet_v3_noniid_fold{fold}.pth"),
        ("MobileNetV3\n(KD Student)", "mobilenet_v3",
         ckpt_root / "kd" / f"mobilenet_v3_distilled_fold{fold}.pth"),
    ]
    checkpoints = [(label, bb, p) for label, bb, p in checkpoints if p.exists()]
    if not checkpoints:
        print("No checkpoints found. Run training first."); return

    results, probs_dict = [], {}
    for label, backbone, ckpt_path in checkpoints:
        print(f"  evaluating {label.replace(chr(10), ' ')} ← {ckpt_path.name}")
        model = build_model(backbone, num_classes=2, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        probs, labels = get_probs_and_labels(model, loader, device)
        # plasma is class 0; ECE function expects (y_true_raw, y_prob)
        y_true_pos = (labels == 0).astype(int)
        ece_val, rel_df = expected_calibration_error(labels, probs, n_bins=args.n_bins)
        results.append({"method": label.replace("\n", " "), "ece": round(ece_val, 5),
                         "fold": fold, "backbone": backbone})
        probs_dict[label] = (probs, y_true_pos, rel_df, ece_val)

    # ── 1. Reliability diagrams ───────────────────────────────────────────────
    n = len(checkpoints)
    ncols = min(n, 3); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for ax, (label, bb, ckpt_path) in zip(axes, checkpoints):
        probs, y_true_pos, rel_df, ece_val = probs_dict[label.replace(" ", "\n")
                                                         if "\n" in label else label]
        df = rel_df[rel_df["count"] > 0]
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect")
        ax.plot(df["confidence"], df["accuracy"], "o-", color="C3",
                label=f"Model (ECE={ece_val:.3f})")
        ax.set_title(label.replace("\n", " "), fontsize=8.5)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Reliability Diagrams — Fold {fold}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability_all_methods.png", dpi=150)
    plt.close(fig)
    print(f"  saved → {out_dir}/reliability_all_methods.png")

    # ── 2. Confidence distribution histograms ─────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for ax, (label, bb, ckpt_path) in zip(axes, checkpoints):
        probs, y_true_pos, rel_df, ece_val = probs_dict[label.replace(" ", "\n")
                                                         if "\n" in label else label]
        ax.hist(probs[y_true_pos == 1], bins=20, alpha=0.6, color="#C44E52", label="Plasma (pos)")
        ax.hist(probs[y_true_pos == 0], bins=20, alpha=0.6, color="#4C72B0", label="Non-plasma (neg)")
        ax.set_title(label.replace("\n", " "), fontsize=8.5)
        ax.set_xlabel("P(plasma)"); ax.set_ylabel("Count")
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Confidence Distribution — Fold {fold}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  saved → {out_dir}/confidence_distributions.png")

    # ── 3. ECE bar chart ──────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_labels = res_df["method"].tolist()
    ece_vals   = res_df["ece"].tolist()
    bars = ax.bar(bar_labels, ece_vals, color=[f"C{i}" for i in range(len(bar_labels))])
    for bar, v in zip(bars, ece_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title(f"Expected Calibration Error — Fold {fold}")
    plt.xticks(rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ece_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  saved → {out_dir}/ece_comparison.png")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = Path(args.results) / "calibration_comparison.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
