"""
analyze_clients.py
------------------
Client heterogeneity analysis for the federated setting. Reads a fold's TRAIN
split and characterises the natural non-IID structure (one client = one patient).

Outputs:
  results/client_stats.csv                  — patient_id, diagnosis, total/plasma/non_plasma, plasma_%
  figures/clients/plasma_percentage.png     — per-patient plasma % (quantity-skew context)
  figures/clients/label_skew_heatmap.png    — plasma vs non_plasma counts per patient
  figures/clients/quantity_skew.png         — total cells per patient (sorted)
  figures/clients/diagnosis_distribution.png

Run:  python analyze_clients.py --config configs/fedavg.yaml --fold 1
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.folds import compute_client_stats  # noqa: E402
from utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fedavg.yaml")
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    fold_id = args.fold or cfg["fold_id"]

    stats = compute_client_stats(cfg["fold_dir"], fold_id)
    res_dir = Path(cfg["results_dir"]); res_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(res_dir / "client_stats.csv", index=False)
    print(stats.to_string(index=False))

    fig_dir = Path(cfg["figures_dir"]) / "clients"; fig_dir.mkdir(parents=True, exist_ok=True)
    pids = stats["patient_id"].astype(str).tolist()

    # 1. plasma percentage per patient
    fig, ax = plt.subplots(figsize=(max(6, len(pids) * 0.4), 4))
    ax.bar(pids, stats["plasma_percentage"], color="#C44E52")
    ax.set_ylabel("Plasma %"); ax.set_xlabel("Patient (client)")
    ax.set_title("Per-client plasma percentage (label skew)")
    plt.xticks(rotation=90)
    fig.tight_layout(); fig.savefig(fig_dir / "plasma_percentage.png", dpi=150); plt.close(fig)

    # 2. label skew heatmap (plasma vs non_plasma)
    mat = np.vstack([stats["plasma_cells"].values, stats["non_plasma_cells"].values])
    fig, ax = plt.subplots(figsize=(max(6, len(pids) * 0.4), 3))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks([0, 1], labels=["plasma", "non_plasma"])
    ax.set_xticks(range(len(pids)), labels=pids, rotation=90)
    ax.set_title("Label distribution per client")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(fig_dir / "label_skew_heatmap.png", dpi=150); plt.close(fig)

    # 3. quantity skew (sorted total cells)
    srt = stats.sort_values("total_cells", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, len(pids) * 0.4), 4))
    ax.bar(srt["patient_id"].astype(str), srt["total_cells"], color="#4C72B0")
    ax.set_ylabel("Total cells"); ax.set_xlabel("Patient (client)")
    ax.set_title("Quantity skew across clients")
    plt.xticks(rotation=90)
    fig.tight_layout(); fig.savefig(fig_dir / "quantity_skew.png", dpi=150); plt.close(fig)

    # 4. diagnosis distribution
    if "diagnosis" in stats.columns:
        counts = stats["diagnosis"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(counts.index.astype(str), counts.values, color="#55A868")
        ax.set_ylabel("# patients"); ax.set_title("Client diagnosis distribution")
        fig.tight_layout(); fig.savefig(fig_dir / "diagnosis_distribution.png", dpi=150); plt.close(fig)

    print(f"\nSaved client_stats.csv and figures to {fig_dir}")


if __name__ == "__main__":
    main()
