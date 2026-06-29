"""
plot_fl_convergence.py
-----------------------
Aggregates fedavg_round_logs.csv / fedprox_round_logs.csv / fedbn_round_logs.csv
across all folds and plots mean ± std convergence curves.

Outputs (in figures/convergence/):
  fedavg_noniid_f1.png       — mean±std val F1 over rounds (FedAvg non-IID, all folds)
  fedavg_noniid_loss.png
  fedprox_noniid_f1.png      — (best mu selection: mu that gave highest mean val F1)
  fedbn_noniid_f1.png
  all_methods_f1.png         — overlay of FedAvg / FedProx / FedBN on same axes

Run:  python plot_fl_convergence.py [--results results] [--figures figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_round_logs(res: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("fedavg", "fedprox", "fedbn"):
        p = res / f"{name}_round_logs.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def _mean_std_curve(df: pd.DataFrame, round_col: str = "round",
                    metric_col: str = "val_f1") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rounds, mean, std) across folds."""
    grouped = df.groupby(round_col)[metric_col].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped[round_col].values, grouped["mean"].values, grouped["std"].values


def _plot_curve(rounds, mean, std, label, ax, color=None):
    ax.plot(rounds, mean, label=label, color=color)
    ax.fill_between(rounds, mean - std, mean + std, alpha=0.2, color=color)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--distribution", default="noniid")
    args = ap.parse_args()
    res = Path(args.results)
    fig_dir = Path(args.figures) / "convergence"
    dist = args.distribution

    logs = _load_round_logs(res)
    if not logs:
        print("[skip] No round_logs CSVs found. Run FL training first.")
        return

    colors = {"fedavg": "C0", "fedprox": "C1", "fedbn": "C2"}
    overlay_data = {}

    for method, df in logs.items():
        df_dist = df[df["distribution"] == dist] if "distribution" in df.columns else df
        if df_dist.empty:
            continue

        if method == "fedprox" and "mu" in df_dist.columns:
            # pick mu with highest mean val_f1 across folds
            best_mu = df_dist.groupby("mu")["val_f1"].mean().idxmax()
            df_dist = df_dist[df_dist["mu"] == best_mu]

        for metric, col in [("f1", "val_f1"), ("loss", "val_loss")]:
            if col not in df_dist.columns:
                continue
            rounds, mean, std = _mean_std_curve(df_dist, "round", col)
            fig, ax = plt.subplots(figsize=(6, 4))
            _plot_curve(rounds, mean, std, label=f"{method} {dist}", ax=ax, color=colors.get(method))
            ax.set_xlabel("Round"); ax.set_ylabel(metric)
            ax.set_title(f"{method.upper()} {dist} — {metric} (mean±std over folds)")
            ax.legend()
            _save(fig, fig_dir / f"{method}_{dist}_{metric}.png")

        # store F1 for overlay
        if "val_f1" in df_dist.columns:
            rounds, mean, std = _mean_std_curve(df_dist, "round", "val_f1")
            overlay_data[method] = (rounds, mean, std)

    # ── Overlay all methods ───────────────────────────────────────────────────
    if len(overlay_data) >= 2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, (rounds, mean, std) in overlay_data.items():
            _plot_curve(rounds, mean, std, label=method.upper(), ax=ax, color=colors.get(method))
        ax.set_xlabel("Round"); ax.set_ylabel("Val F1")
        ax.set_title(f"FL convergence comparison — {dist} (mean±std over folds)")
        ax.legend()
        _save(fig, fig_dir / f"all_methods_{dist}_f1.png")

    print(f"Convergence plots saved to {fig_dir}/")


if __name__ == "__main__":
    main()
