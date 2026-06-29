"""
run_rounds_ablation.py
-----------------------
Ablation on number of FL communication rounds r ∈ {10, 20, 50, 100}.

Uses existing round_logs.csv (already produced by FL training) to extract
performance at different round cutoffs — no retraining needed.

If round_logs CSVs have fewer rows than max_rounds, skips those values.

Outputs:
  results/rounds_ablation.csv
  figures/ablation/rounds_vs_f1.png
  figures/ablation/rounds_convergence_threshold.png  — rounds to reach X% of best

Run:
  python run_rounds_ablation.py [--results results] [--rounds 10 20 50 100]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    ("fedavg",  "FedAvg",  "C0"),
    ("fedprox", "FedProx", "C1"),
    ("fedbn",   "FedBN",   "C2"),
]


def _best_f1_at_round(df: pd.DataFrame, round_cutoff: int, dist: str = "noniid") -> float:
    sub = df[(df["distribution"] == dist) & (df["round"] <= round_cutoff)] \
        if "distribution" in df.columns else df[df["round"] <= round_cutoff]
    if sub.empty:
        return np.nan
    f1_col = "val_f1" if "val_f1" in sub.columns else "f1"
    return float(sub[f1_col].max())


def _rounds_to_threshold(df: pd.DataFrame, pct: float, dist: str = "noniid") -> int:
    sub = df[(df["distribution"] == dist)] if "distribution" in df.columns else df
    if sub.empty:
        return -1
    f1_col = "val_f1" if "val_f1" in sub.columns else "f1"
    best = float(sub[f1_col].max())
    target = best * pct
    hit = sub[sub[f1_col] >= target]
    return int(hit["round"].min()) if not hit.empty else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--rounds",  type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--dist",    default="noniid")
    args = ap.parse_args()
    res = Path(args.results)
    fig_dir = Path(args.figures) / "ablation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_dfs = {}
    for key, label, _ in METHODS:
        p = res / f"{key}_round_logs.csv"
        if p.exists():
            log_dfs[key] = pd.read_csv(p)
            print(f"  loaded {p.name} ({len(log_dfs[key])} rows)")
        else:
            print(f"  [skip] {p.name} not found")

    if not log_dfs:
        print("No round_logs CSVs found. Run FL training first."); return

    # ── Table: F1 at each round cutoff ───────────────────────────────────────
    rows = []
    for key, label, _ in METHODS:
        if key not in log_dfs:
            continue
        df = log_dfs[key]
        max_round = int(df["round"].max())
        for r in args.rounds:
            if r > max_round:
                continue
            f1 = _best_f1_at_round(df, r, args.dist)
            rows.append({"method": label, "rounds": r, "best_val_f1": round(f1, 5)})

    if not rows:
        print("No data to report."); return

    abl_df = pd.DataFrame(rows)
    out_csv = res / "rounds_ablation.csv"
    abl_df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print(abl_df.to_string(index=False))

    # ── Plot 1: rounds vs F1 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label, color in METHODS:
        sub = abl_df[abl_df["method"] == label].sort_values("rounds")
        if sub.empty:
            continue
        ax.plot(sub["rounds"], sub["best_val_f1"], "o-", label=label, color=color)
    ax.set_xlabel("Number of rounds"); ax.set_ylabel("Best val F1 (cumulative max)")
    ax.set_title(f"Performance vs Training Rounds ({args.dist})")
    ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / "rounds_vs_f1.png", dpi=150); plt.close(fig)
    print(f"  saved → {fig_dir}/rounds_vs_f1.png")

    # ── Plot 2: rounds to reach 90% / 95% / 99% of best F1 ───────────────────
    thresholds = [0.90, 0.95, 0.99]
    thresh_rows = []
    for key, label, _ in METHODS:
        if key not in log_dfs:
            continue
        for thr in thresholds:
            r = _rounds_to_threshold(log_dfs[key], thr, args.dist)
            thresh_rows.append({"method": label, "threshold": f"{int(thr*100)}%", "rounds": r})

    if thresh_rows:
        tdf = pd.DataFrame(thresh_rows)
        tdf_pivot = tdf.pivot(index="threshold", columns="method", values="rounds")
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(tdf_pivot))
        w = 0.25
        for i, (col, color) in enumerate([(m[1], m[2]) for m in METHODS if m[1] in tdf_pivot.columns]):
            offset = (i - len(tdf_pivot.columns) / 2 + 0.5) * w
            vals = tdf_pivot[col].values
            valid = vals > 0
            ax.bar(x[valid] + offset, vals[valid], width=w * 0.9, label=col, color=color)
        ax.set_xticks(x); ax.set_xticklabels(tdf_pivot.index)
        ax.set_xlabel("Target F1 (% of best)"); ax.set_ylabel("Rounds to reach threshold")
        ax.set_title("Convergence Speed (rounds to reach X% of best F1)")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "rounds_convergence_threshold.png", dpi=150)
        plt.close(fig)
        print(f"  saved → {fig_dir}/rounds_convergence_threshold.png")


if __name__ == "__main__":
    main()
