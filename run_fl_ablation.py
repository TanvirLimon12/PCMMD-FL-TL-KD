"""
run_fl_ablation.py
-------------------
Ablation study for FL hyperparameters:
  • client_fraction C ∈ {0.5, 0.75, 1.0}   — partial participation
  • local_epochs    E ∈ {1, 3, 5, 10}       — client drift

Runs FedAvg (default) or FedBN for fold 1. Each (C, E) combination gets one run.
Results appended to results/fl_ablation.csv.
Figures saved to figures/ablation/.

Run:
  python run_fl_ablation.py [--method fedavg] [--fold 1]
  python run_fl_ablation.py --method fedbn --c_fracs 0.5 0.75 1.0 --local_epochs 1 3 5 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


METHOD_SCRIPTS = {
    "fedavg":  ("train_fedavg.py",  "configs/fedavg.yaml"),
    "fedbn":   ("train_fedbn.py",   "configs/fedbn.yaml"),
    "fedprox": ("train_fedprox.py", "configs/fedprox.yaml"),
}

RESULT_COLS = {
    "fedavg":  "fedavg_results.csv",
    "fedbn":   "fedbn_results.csv",
    "fedprox": "fedprox_results.csv",
}


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _latest_row(results_dir: str, method: str, fold: int, dist: str,
                c_frac: float, local_epochs: int) -> dict | None:
    p = Path(results_dir) / RESULT_COLS[method]
    if not p.exists():
        return None
    df = pd.read_csv(p)
    mask = (df["fold"] == fold) & (df["distribution"] == dist)
    if "client_fraction" in df.columns:
        mask &= (df["client_fraction"] - c_frac).abs() < 1e-6
    if "local_epochs" in df.columns:
        mask &= df["local_epochs"] == local_epochs
    sub = df[mask]
    return sub.iloc[-1].to_dict() if not sub.empty else None


def _run_one(script: str, base_cfg: dict, c_frac: float, local_epochs: int,
             fold: int, dist: str, results_dir: str) -> None:
    override = dict(base_cfg)
    override["client_fraction"] = c_frac
    override["local_epochs"]    = local_epochs
    override["fold_id"]         = fold
    override["distribution"]    = dist
    override["results_dir"]     = results_dir
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tf:
        yaml.dump(override, tf, default_flow_style=False)
        tmp = tf.name
    ret = subprocess.run([sys.executable, script, "--config", tmp])
    Path(tmp).unlink(missing_ok=True)
    if ret.returncode != 0:
        print(f"  [warn] {script} C={c_frac} E={local_epochs} exited {ret.returncode}")


def _plot_ablation(df: pd.DataFrame, fig_dir: Path, method: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    f1_col = "besttest_f1" if "besttest_f1" in df.columns else "f1"

    # ── C-fraction vs F1 (one line per E) ────────────────────────────────────
    if "client_fraction" in df.columns and "local_epochs" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, (x_col, grp_col, xlabel, title) in zip(axes, [
            ("client_fraction", "local_epochs",
             "Client fraction C", "Participation rate ablation"),
            ("local_epochs", "client_fraction",
             "Local epochs E", "Local epochs ablation"),
        ]):
            for grp_val, sub in df.groupby(grp_col):
                sub = sub.sort_values(x_col)
                ax.plot(sub[x_col].astype(str), sub[f1_col],
                        "o-", label=f"{grp_col}={grp_val}")
            ax.set_xlabel(xlabel); ax.set_ylabel("Best-test F1")
            ax.set_title(f"{method.upper()} — {title}")
            ax.legend(fontsize=8)
        fig.suptitle(f"FL Hyperparameter Ablation — {method.upper()}", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{method}_ablation.png", dpi=150)
        plt.close(fig)
        print(f"  saved → {fig_dir / f'{method}_ablation.png'}")

    # ── Heatmap C × E → F1 ───────────────────────────────────────────────────
    if "client_fraction" in df.columns and "local_epochs" in df.columns:
        pivot = df.pivot_table(index="local_epochs", columns="client_fraction",
                               values=f1_col, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"C={v:.2f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"E={v}" for v in pivot.index])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im)
        ax.set_title(f"{method.upper()} — F1 vs (C, E)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{method}_ablation_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  saved → {fig_dir / f'{method}_ablation_heatmap.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="fedavg", choices=list(METHOD_SCRIPTS))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--distribution", default="noniid")
    ap.add_argument("--c_fracs", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--local_epochs", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip (C,E) combinations already in results CSV")
    args = ap.parse_args()

    script, base_cfg_path = METHOD_SCRIPTS[args.method]
    base_cfg = _load_yaml(base_cfg_path)
    sweep_rows = []

    total = len(args.c_fracs) * len(args.local_epochs)
    done = 0
    for c in args.c_fracs:
        for e in args.local_epochs:
            done += 1
            print(f"\n[{done}/{total}] {args.method} C={c} E={e} fold={args.fold}")
            if args.skip_existing:
                row = _latest_row(args.results, args.method, args.fold,
                                  args.distribution, c, e)
                if row:
                    print("  [skip] already in results CSV")
                    sweep_rows.append({**row, "client_fraction": c, "local_epochs": e})
                    continue
            _run_one(script, base_cfg, c, e, args.fold, args.distribution, args.results)
            row = _latest_row(args.results, args.method, args.fold,
                              args.distribution, c, e)
            if row:
                sweep_rows.append({**row, "client_fraction": c, "local_epochs": e})

    if not sweep_rows:
        print("No results collected.")
        return

    df = pd.DataFrame(sweep_rows)
    out_csv = Path(args.results) / "fl_ablation.csv"
    # merge with any existing rows
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["method", "distribution", "fold", "client_fraction", "local_epochs"],
            keep="last"
        )
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    _plot_ablation(df[df["method"] == args.method],
                   Path(args.figures) / "ablation", args.method)


if __name__ == "__main__":
    main()
