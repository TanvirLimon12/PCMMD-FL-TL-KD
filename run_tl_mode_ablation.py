"""
run_tl_mode_ablation.py
------------------------
Ablation of transfer-learning fine-tuning strategy:
  frozen  — backbone frozen, only classifier head trained
  partial — first 70% of layers frozen, rest + head trained
  full    — all parameters trained (default)

Runs all 3 backbones × 3 modes via train_centralized.py.
Results saved to results/tl_mode_ablation.csv.
Figure saved to figures/ablation/tl_mode_comparison.png.

Run:
  python run_tl_mode_ablation.py [--fold 1] [--backbones resnet50 efficientnet_b0 mobilenet_v3]
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

MODES = ["frozen", "partial", "full"]
BACKBONE_DISPLAY = {
    "resnet50":        "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3":    "MobileNetV3",
}


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_row(results_dir: str, backbone: str, fold: int, mode: str) -> dict | None:
    p = Path(results_dir) / "centralized_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    mask = (df["backbone"] == backbone) & (df["fold"] == fold)
    if "finetune_mode" in df.columns:
        mask &= df["finetune_mode"] == mode
    sub = df[mask]
    return sub.iloc[-1].to_dict() if not sub.empty else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet50", "efficientnet_b0", "mobilenet_v3"])
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--config", default="configs/centralized.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()

    base_cfg = _load_yaml(args.config)
    rows = []
    total = len(args.backbones) * len(args.modes)
    done = 0

    for backbone in args.backbones:
        for mode in args.modes:
            done += 1
            print(f"\n[{done}/{total}] backbone={backbone} mode={mode} fold={args.fold}")
            existing = _get_row(args.results, backbone, args.fold, mode)
            if existing:
                print("  [skip] already in centralized_results.csv")
                rows.append({**existing, "finetune_mode": mode})
                continue

            override = dict(base_cfg)
            override["backbone"]      = backbone
            override["finetune_mode"] = mode
            override["folds"]         = [args.fold]
            override["results_dir"]   = args.results

            with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tf:
                yaml.dump(override, tf, default_flow_style=False)
                tmp = tf.name
            ret = subprocess.run([sys.executable, "train_centralized.py",
                                  "--config", tmp, "--backbone", backbone])
            Path(tmp).unlink(missing_ok=True)
            if ret.returncode != 0:
                print(f"  [warn] exited {ret.returncode}")
                continue
            row = _get_row(args.results, backbone, args.fold, mode)
            if row:
                rows.append({**row, "finetune_mode": mode})

    if not rows:
        print("No results collected.")
        return

    df = pd.DataFrame(rows)
    out_csv = Path(args.results) / "tl_mode_ablation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── Figure: grouped bar backbone × mode → F1 ────────────────────────────
    fig_dir = Path(args.figures) / "ablation"; fig_dir.mkdir(parents=True, exist_ok=True)
    f1_col = "f1" if "f1" in df.columns else ([c for c in df.columns if "f1" in c] or ["f1"])[0]

    backbones_present = [b for b in args.backbones if b in df["backbone"].values]
    modes_present     = [m for m in MODES if m in df.get("finetune_mode", pd.Series([])).values]
    if not modes_present:
        modes_present = MODES

    x = np.arange(len(backbones_present))
    width = 0.25
    mode_colors = {"frozen": "#4C72B0", "partial": "#DD8452", "full": "#55A868"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mode in enumerate(modes_present):
        vals = []
        for bb in backbones_present:
            sub = df[(df["backbone"] == bb)]
            if "finetune_mode" in sub.columns:
                sub = sub[sub["finetune_mode"] == mode]
            vals.append(float(sub[f1_col].mean()) if not sub.empty else 0.0)
        offset = (i - len(modes_present) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=mode, color=mode_colors.get(mode, f"C{i}"))
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONE_DISPLAY.get(b, b) for b in backbones_present])
    ax.set_ylabel("Test F1"); ax.set_ylim(0.75, 1.0)
    ax.set_title("Transfer Learning Strategy Ablation\n(frozen / partial / full fine-tuning)")
    ax.legend(title="Finetune mode", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "tl_mode_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {fig_dir}/tl_mode_comparison.png")

    print("\n" + df[["backbone", "finetune_mode", f1_col]].to_string(index=False))


if __name__ == "__main__":
    main()
