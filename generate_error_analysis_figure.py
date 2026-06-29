"""
generate_error_analysis_figure.py
----------------------------------
Aggregates error_*_fold*_breakdown.csv files into summary figures:
1. Per-fold TP/TN/FP/FN grouped bar per backbone
2. Aggregate FP/FN cross-backbone comparison
3. Plasma false-negative rate per backbone
"""
from __future__ import annotations
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}
BACKBONE_DISPLAY = {
    "effnet":    "EfficientNet-B0",
    "mobilenet": "MobileNetV3",
    "resnet50":  "ResNet50",
}


def load_all_breakdowns(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("error_*_fold*_breakdown.csv")):
        m = re.match(r"error_(\w+)_fold(\d+)_breakdown\.csv", p.name)
        if not m:
            continue
        tag, fold = m.group(1), int(m.group(2))
        df = pd.read_csv(p)
        row = df.iloc[0].to_dict()
        row["backbone"] = BACKBONE_DISPLAY.get(tag, tag)
        row["fold"] = fold
        rows.append(row)
    return pd.DataFrame(rows)


def plot_per_fold_bar(df: pd.DataFrame, out_path: Path) -> None:
    backbones = df["backbone"].unique().tolist()
    fig, axes = plt.subplots(1, len(backbones), figsize=(5 * len(backbones), 4.5), sharey=False)
    if len(backbones) == 1:
        axes = [axes]
    for ax, bb in zip(axes, backbones):
        sub = df[df["backbone"] == bb].sort_values("fold")
        x = np.arange(len(sub))
        w = 0.2
        for i, col in enumerate(["TP", "TN", "FP", "FN"]):
            ax.bar(x + i * w, sub[col], w, label=col, color=COLORS[col], alpha=0.85)
        ax.set_xticks(x + 1.5 * w)
        ax.set_xticklabels([f"F{r}" for r in sub["fold"]])
        ax.set_title(bb, fontsize=10)
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
    fig.suptitle("Per-fold TP/TN/FP/FN by Backbone", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_aggregate_comparison(df: pd.DataFrame, out_path: Path) -> None:
    agg = df.groupby("backbone")[["TP", "TN", "FP", "FN"]].sum().reset_index()
    agg["n"] = agg[["TP", "TN", "FP", "FN"]].sum(axis=1)
    agg["FP_rate"] = agg["FP"] / agg["n"] * 100
    agg["FN_rate"] = agg["FN"] / agg["n"] * 100
    agg["err_rate"] = (agg["FP"] + agg["FN"]) / agg["n"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Stacked TP/TN/FP/FN totals
    x = np.arange(len(agg))
    bottoms = np.zeros(len(agg))
    for col in ["TP", "TN", "FP", "FN"]:
        axes[0].bar(x, agg[col], bottom=bottoms, label=col, color=COLORS[col], alpha=0.85)
        bottoms += agg[col].values
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(agg["backbone"], rotation=15, ha="right")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Total Predictions")
    axes[0].legend()

    # FP/FN rates
    w = 0.35
    axes[1].bar(x - w / 2, agg["FP_rate"], w, label="FP rate", color=COLORS["FP"], alpha=0.85)
    axes[1].bar(x + w / 2, agg["FN_rate"], w, label="FN rate", color=COLORS["FN"], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(agg["backbone"], rotation=15, ha="right")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("FP / FN Rates")
    axes[1].legend()

    # Overall error rate
    axes[2].bar(x, agg["err_rate"], color="#9b59b6", alpha=0.85)
    for i, v in enumerate(agg["err_rate"]):
        axes[2].text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(agg["backbone"], rotation=15, ha="right")
    axes[2].set_ylabel("Error Rate (%)")
    axes[2].set_title("Overall Error Rate")

    fig.suptitle("Aggregate Error Analysis Across Backbones", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fn_analysis(df: pd.DataFrame, out_path: Path) -> None:
    """Focus on plasma cell false negatives (clinically important)."""
    agg = df.groupby("backbone")[["FN", "TP"]].sum().reset_index()
    agg["sensitivity"] = agg["TP"] / (agg["TP"] + agg["FN"]) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(agg))
    bars = ax.bar(x, agg["sensitivity"], color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(99, color="red", ls="--", lw=1.5, label="99% threshold")
    ax.set_ylim(96, 100.5)
    for bar, val in zip(bars, agg["sensitivity"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["backbone"], rotation=15, ha="right")
    ax.set_ylabel("Plasma Cell Sensitivity (%)")
    ax.set_title("Sensitivity (Recall) per Backbone\n(Plasma Cell Detection)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    results_dir = Path("results")
    out_dir = Path("figures/errors")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_breakdowns(results_dir)
    if df.empty:
        print("No breakdown CSVs found in results/")
        return

    print(f"Loaded {len(df)} fold×backbone results")
    print(df.groupby("backbone")[["TP", "TN", "FP", "FN"]].sum())

    plot_per_fold_bar(df, out_dir / "error_per_fold_bar.png")
    plot_aggregate_comparison(df, out_dir / "error_aggregate_comparison.png")
    plot_fn_analysis(df, out_dir / "error_sensitivity_comparison.png")

    # Save summary table
    agg = df.groupby("backbone")[["TP", "TN", "FP", "FN"]].sum()
    agg["sensitivity"] = agg["TP"] / (agg["TP"] + agg["FN"])
    agg["specificity"] = agg["TN"] / (agg["TN"] + agg["FP"])
    agg["error_rate"] = (agg["FP"] + agg["FN"]) / (agg[["TP", "TN", "FP", "FN"]].sum(axis=1))
    agg.to_csv(results_dir / "error_aggregate_summary.csv")
    print(f"Saved: {results_dir}/error_aggregate_summary.csv")


if __name__ == "__main__":
    main()
