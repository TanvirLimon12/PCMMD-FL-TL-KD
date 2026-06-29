"""
plot_communication_analysis.py
-------------------------------
Reads results/communication_analysis.csv (written by train_fedavg/fedprox/fedbn)
and produces:

  figures/comm/comm_vs_f1.png          — scatter: cumulative MB vs best-test F1
  figures/comm/rounds_vs_comm.png      — line: cumulative MB per round per method
  figures/comm/comm_efficiency_table.csv — summary table (method, total_comm_mb, best_f1)

Also reads fedavg_results.csv, fedprox_results.csv, fedbn_results.csv to match
best-test F1 per method/distribution.

Run:  python plot_communication_analysis.py [--results results] [--figures figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import plot_comm_vs_accuracy


def _load_results(res: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("fedavg_results", "fedprox_results", "fedbn_results"):
        p = res / f"{name}.csv"
        if p.exists():
            out[name.replace("_results", "")] = pd.read_csv(p)
    return out


def _best_f1_per_method_dist(dfs: dict[str, pd.DataFrame]) -> dict[tuple, float]:
    """Return {(method, distribution): best_test_f1} averaged over folds."""
    mapping = {}
    for method, df in dfs.items():
        if "fedprox" in method and "mu" in df.columns:
            # pick best mu per fold, then average
            ni = df[df["distribution"] == "noniid"]
            if len(ni):
                best_per_fold = ni.loc[ni.groupby("fold")["best_val_f1"].idxmax()]
                col = "besttest_f1" if "besttest_f1" in best_per_fold.columns else "f1"
                mapping[(method, "noniid")] = float(best_per_fold[col].mean())
        else:
            col = "besttest_f1" if "besttest_f1" in df.columns else "f1"
            for dist, sub in df.groupby("distribution"):
                mapping[(method, dist)] = float(sub[col].mean())
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()
    res, fig_root = Path(args.results), Path(args.figures)
    comm_path = res / "communication_analysis.csv"
    if not comm_path.exists():
        print(f"[skip] {comm_path} not found — run FL training first.")
        return

    comm = pd.read_csv(comm_path)
    result_dfs = _load_results(res)
    f1_map = _best_f1_per_method_dist(result_dfs)

    fig_dir = fig_root / "comm"; fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Rounds vs cumulative comm per method/dist ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (method, dist), sub in comm.groupby(["method", "distribution"]):
        sub = sub.sort_values("round")
        ax.plot(sub["round"], sub["cumulative_comm_mb"], label=f"{method}-{dist}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative communication (MB)")
    ax.set_title("Cumulative communication cost per round")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "rounds_vs_comm.png", dpi=150)
    plt.close(fig)

    # ── 2. Comm vs accuracy scatter ───────────────────────────────────────────
    summary_rows = []
    for (method, dist), sub in comm.groupby(["method", "distribution"]):
        total_mb = float(sub["cumulative_comm_mb"].max())
        f1 = f1_map.get((method, dist), np.nan)
        summary_rows.append({"method": method, "distribution": dist,
                              "total_comm_mb": round(total_mb, 2), "best_f1": round(f1, 4)})
    summary = pd.DataFrame(summary_rows).dropna(subset=["best_f1"])
    summary.to_csv(fig_dir / "comm_efficiency_table.csv", index=False)

    if len(summary):
        labels = [f"{r['method']}-{r['distribution']}" for _, r in summary.iterrows()]
        plot_comm_vs_accuracy(
            labels=labels,
            comm_mb=summary["total_comm_mb"].tolist(),
            f1_scores=summary["best_f1"].tolist(),
            out_path=fig_dir / "comm_vs_f1.png",
            title="Communication cost vs best-test F1",
        )

    print(f"Saved figures to {fig_dir}/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
