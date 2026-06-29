"""
run_dirichlet_sweep.py
-----------------------
Sweeps Dirichlet concentration α ∈ {0.1, 0.5, 1.0} for FedAvg and FedBN
(fold 1 only, one run per alpha — for ablation / heterogeneity analysis).

For each (method, alpha):
  1. Writes a temporary override config
  2. Invokes the training script
  3. Collects final result from results/{method}_results.csv
  4. Appends a row to results/dirichlet_sweep.csv

Outputs:
  results/dirichlet_sweep.csv
  figures/dirichlet/alpha_vs_f1.png   — line: α on x, best-test F1 on y per method

Run:  python run_dirichlet_sweep.py [--fold 1] [--alphas 0.1 0.5 1.0]
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
import pandas as pd
import yaml


BASE_CONFIGS = {
    "fedavg": "configs/fedavg.yaml",
    "fedbn":  "configs/fedbn.yaml",
}

SCRIPTS = {
    "fedavg": "train_fedavg.py",
    "fedbn":  "train_fedbn.py",
}


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _write_yaml(data: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def _latest_result(results_dir: str, method: str, fold: int, distribution: str) -> dict | None:
    p = Path(results_dir) / f"{method}_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    row = df[(df["fold"] == fold) & (df["distribution"] == distribution)]
    if row.empty:
        return None
    return row.iloc[-1].to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()

    sweep_rows = []

    for method, base_cfg_path in BASE_CONFIGS.items():
        cfg = _load_yaml(base_cfg_path)
        for alpha in args.alphas:
            print(f"\n{'='*60}")
            print(f"  method={method}  alpha={alpha}  fold={args.fold}")
            print(f"{'='*60}")

            override = dict(cfg)
            override["partition"] = "dirichlet"
            override["dirichlet_alpha"] = alpha
            override["fold_id"] = args.fold
            override["distribution"] = "noniid"
            # Tag the distribution so results are stored separately
            override["distribution"] = f"dirichlet_a{alpha}"

            with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tf:
                yaml.dump(override, tf, default_flow_style=False)
                tmp_cfg = tf.name

            ret = subprocess.run(
                [sys.executable, SCRIPTS[method], "--config", tmp_cfg],
                capture_output=False
            )
            Path(tmp_cfg).unlink(missing_ok=True)

            if ret.returncode != 0:
                print(f"[warn] {method} alpha={alpha} exited with code {ret.returncode}")
                continue

            row = _latest_result(args.results, method, args.fold,
                                  f"dirichlet_a{alpha}")
            if row:
                row["dirichlet_alpha"] = alpha
                sweep_rows.append(row)

    if not sweep_rows:
        print("No sweep results collected.")
        return

    out_csv = Path(args.results) / "dirichlet_sweep.csv"
    pd.DataFrame(sweep_rows).to_csv(out_csv, index=False)
    print(f"\nSaved sweep results → {out_csv}")

    # ── Plot alpha vs F1 per method ──────────────────────────────────────────
    df = pd.DataFrame(sweep_rows)
    f1_col = "besttest_f1" if "besttest_f1" in df.columns else "f1"
    fig_dir = Path(args.figures) / "dirichlet"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("dirichlet_alpha")
        ax.plot(sub["dirichlet_alpha"], sub[f1_col], marker="o", label=method.upper())
    ax.set_xlabel("Dirichlet α (lower = more heterogeneous)")
    ax.set_ylabel("Best-test F1")
    ax.set_title("Effect of data heterogeneity (Dirichlet α) on FL performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "alpha_vs_f1.png", dpi=150)
    plt.close(fig)
    print(f"Saved figure → {fig_dir}/alpha_vs_f1.png")


if __name__ == "__main__":
    main()
