"""
error_analysis.py
-----------------
Error analysis for the best centralized model (Abrar §A4).

Loads a checkpoint, scores the fold TEST split, and saves:
  results/error_<tag>_fold<k>_misclassified.csv  — image_id, patient_id, true, pred, confidence, fold
  results/error_<tag>_fold<k>_breakdown.csv       — TP/TN/FP/FN counts
  figures/errors/<tag>_fold<k>_panel.png          — example crops for TP/TN/FP/FN

Clinically, plasma-cell false negatives (missed plasma) matter most, so FN rows
are sorted by ascending confidence (most confidently wrong first).

Run:
  python error_analysis.py --config configs/centralized.yaml \
      --weights checkpoints/centralized/efficientnet_b0_fold1.pth --fold 1 --tag effnet
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import IDX_TO_CLASS, build_loader  # noqa: E402
from data.transforms import get_val_transforms  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_config, set_seed  # noqa: E402

POSITIVE = 0  # plasma is the positive class


@torch.no_grad()
def collect(model, loader, device):
    """Return per-sample records with prediction, confidence, image path."""
    model.eval()
    recs = []
    for batch in loader:
        imgs = batch[0].to(device)
        lbls = batch[1].numpy()
        pids = batch[2] if len(batch) > 2 else ["?"] * len(lbls)
        paths = batch[3] if len(batch) > 3 else ["?"] * len(lbls)
        prob = F.softmax(model(imgs), dim=1).cpu().numpy()
        pred = prob.argmax(1)
        for i in range(len(lbls)):
            recs.append({"patient_id": str(pids[i]), "path": str(paths[i]),
                         "true": int(lbls[i]), "pred": int(pred[i]),
                         "confidence": round(float(prob[i, pred[i]]), 5),
                         "prob_plasma": round(float(prob[i, POSITIVE]), 5)})
    return pd.DataFrame(recs)


def _bucket(df: pd.DataFrame) -> dict:
    """TP/TN/FP/FN where positive=plasma(0)."""
    return {
        "TP": df[(df.true == POSITIVE) & (df.pred == POSITIVE)],
        "TN": df[(df.true != POSITIVE) & (df.pred != POSITIVE)],
        "FP": df[(df.true != POSITIVE) & (df.pred == POSITIVE)],
        "FN": df[(df.true == POSITIVE) & (df.pred != POSITIVE)],
    }


def _panel(buckets: dict, out_path: Path, n: int = 4) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cats = ["TP", "TN", "FP", "FN"]
    fig, axes = plt.subplots(len(cats), n, figsize=(n * 2.2, len(cats) * 2.4))
    for r, cat in enumerate(cats):
        sub = buckets[cat]
        sub = sub.sort_values("confidence") if cat == "FN" else sub  # worst FN first
        for c in range(n):
            ax = axes[r, c] if len(cats) > 1 else axes[c]
            ax.axis("off")
            if c == 0:
                ax.set_title(cat, loc="left", fontsize=11, fontweight="bold")
            if c < len(sub):
                row = sub.iloc[c]
                try:
                    ax.imshow(Image.open(row["path"]).convert("RGB"))
                    ax.set_title(f"conf={row['confidence']:.2f}", fontsize=8)
                except Exception:
                    ax.text(0.5, 0.5, "img?", ha="center", va="center")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/centralized.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--panel", action="store_true", help="Also render TP/TN/FP/FN image panel")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed()
    device = get_device()
    fold_id = args.fold or cfg["fold_id"]
    backbone = args.backbone or cfg["backbone"]
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    csv_path = Path(cfg["fold_dir"]) / f"fold_{fold_id}.csv"
    loader = build_loader(csv_path=csv_path, split="test", transform=get_val_transforms(),
                          batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                          root_dir=cfg["root_dir"], image_root=cfg["image_root"], return_meta=True)
    model = build_model(backbone, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    df = collect(model, loader, device)
    df["image_id"] = df["path"].map(lambda p: Path(p).name)
    df["true_label"] = df["true"].map(IDX_TO_CLASS)
    df["pred_label"] = df["pred"].map(IDX_TO_CLASS)
    df["fold"] = fold_id

    res = Path(cfg["results_dir"]); res.mkdir(parents=True, exist_ok=True)
    stem = f"error_{args.tag}_fold{fold_id}"
    mis = df[df.true != df.pred].sort_values(["true", "confidence"], ascending=[True, False])
    mis[["image_id", "patient_id", "true_label", "pred_label", "confidence", "fold"]].to_csv(
        res / f"{stem}_misclassified.csv", index=False)

    buckets = _bucket(df)
    pd.DataFrame([{k: len(v) for k, v in buckets.items()}]).to_csv(
        res / f"{stem}_breakdown.csv", index=False)

    if args.panel:
        _panel(buckets, Path(cfg["figures_dir"]) / "errors" / f"{stem}_panel.png")

    print(f"TP={len(buckets['TP'])} TN={len(buckets['TN'])} "
          f"FP={len(buckets['FP'])} FN={len(buckets['FN'])} "
          f"(plasma false-negatives={len(buckets['FN'])})")
    print(f"Saved: {res / (stem + '_misclassified.csv')}")


if __name__ == "__main__":
    main()
