"""
Generate a publication-quality data-efficiency figure from fewshot_results.csv.
Run after train_fewshot.py completes.
Output: figures/fewshot/data_efficiency_combined.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results/fewshot_results.csv")
OUT     = Path("figures/fewshot/data_efficiency_combined.png")

BACKBONE_LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3":    "MobileNetV3",
}
COLORS = {
    "efficientnet_b0": "#4C72B0",
    "mobilenet_v3":    "#DD8452",
}
MARKERS = {
    "efficientnet_b0": "o",
    "mobilenet_v3":    "s",
}


def main():
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} not found. Run train_fewshot.py first."); return

    df = pd.read_csv(RESULTS)
    # handle column name variants
    pct_col = "data_pct" if "data_pct" in df.columns else "pct"
    pcts = sorted(df[pct_col].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, metric, label in zip(axes, ["f1", "roc_auc"], ["F1 Score", "ROC-AUC"]):
        for bb in ["efficientnet_b0", "mobilenet_v3"]:
            sub = df[df["backbone"] == bb].sort_values(pct_col)
            vals = sub[metric].values
            ax.plot(sub[pct_col], vals,
                    marker=MARKERS[bb], color=COLORS[bb],
                    label=BACKBONE_LABELS[bb], linewidth=2, markersize=7)
            # add value labels
            for x, y in zip(sub[pct_col], vals):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7)
        ax.set_xlabel("Labeled Training Data (%)", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"Data Efficiency — {label}", fontsize=13)
        ax.set_xticks(pcts)
        ax.set_xticklabels([f"{p}%" for p in pcts])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.02)

    fig.suptitle("Few-Shot Learning: Performance vs. Labeled Data Fraction\n(Fold 1, Full Fine-Tuning)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
