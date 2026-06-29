"""
train_fewshot.py
----------------
Data-efficiency / few-shot experiments (Abrar §A5, proposal E6 / RQ4).

Trains a backbone on each FROZEN few-shot pool (fewshot_5/10/20/50/100.csv) and
evaluates on a fold's TEST split. The same test fold is used for every percentage.

LEAKAGE-SAFE: the few-shot loader drops any pool row whose image identity (md5,
else basename) collides with the test fold (handled in data.folds.get_fewshot_loader).
Model selection uses the patient-disjoint VAL set; TEST is scored once per setting.

Outputs:
  results/fewshot_results.csv          — model × data% × fold × seed (Acc/F1/ROC-AUC/PR-AUC/plasma-recall)
  results/fewshot_summary.csv          — mean ± std across folds per (model, data%), canonical seed only
  figures/fewshot/<model>_curve.png    — F1 & PR-AUC vs labeled-data %, canonical seed only
  checkpoints/fewshot/<model>_p<pct>_fold<k>[_seed<s>].pth

--seed varies model init / batch order only — the data% subsamples are frozen in
data/eda/fewshot_*.csv. Useful for checking whether a given data% result is stable.

Run:  python train_fewshot.py --config configs/fewshot.yaml [--seed 42]
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
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.folds import get_fewshot_loader, get_fold_loaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    SEED, collect_predictions, compute_all_metrics, get_device, load_config,
    save_config_snapshot, set_seed, setup_logging, summarise_folds,
)

METRIC_COLS = ["accuracy", "f1", "roc_auc", "pr_auc", "sensitivity"]  # sensitivity = plasma recall


def _metrics(model, loader, device):
    yt, yp, pr, _ = collect_predictions(model, loader, device)
    return compute_all_metrics(yt, yp, pr)


def train_one(cfg, device, logger, backbone, pct, fold_id, seed, train_loader, monitor, test_loader) -> dict:
    ckpt_dir = Path(cfg["ckpt_dir"]) / "fewshot"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if seed == SEED else f"_seed{seed}"
    ckpt = ckpt_dir / f"{backbone}_p{pct}_fold{fold_id}{suffix}.pth"
    model = build_model(backbone, num_classes=2, pretrained=cfg["pretrained"],
                        finetune_mode=cfg.get("finetune_mode", "full")).to(device)
    if ckpt.exists():
        logger.info("  [%s p%s f%d seed%d] checkpoint exists — skip training, evaluating",
                    backbone, pct, fold_id, seed)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        tm = _metrics(model, test_loader, device)
        return {"backbone": backbone, "data_pct": pct, "fold": fold_id, "seed": seed,
                **{k: round(tm[k], 5) for k in METRIC_COLS}}
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_f1, patience_ctr = -1.0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for batch in train_loader:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        vm = _metrics(model, monitor, device)
        if vm["f1"] > best_f1:
            best_f1, patience_ctr = vm["f1"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                break
    logger.info("  [%s p%s f%d seed%d] best_val_f1=%.4f", backbone, pct, fold_id, seed, best_f1)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    tm = _metrics(model, test_loader, device)
    return {"backbone": backbone, "data_pct": pct, "fold": fold_id, "seed": seed,
            **{k: round(tm[k], 5) for k in METRIC_COLS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fewshot.yaml")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="Training-process seed (model init, batch order). The few-shot "
                         "subsamples themselves are frozen in data/eda/fewshot_*.csv and do "
                         "not change with this flag.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    backbones = cfg.get("backbones", ["efficientnet_b0", "mobilenet_v3"])
    pcts = cfg.get("data_pcts", [5, 10, 20, 50, 100])
    folds = cfg["folds"]

    set_seed(args.seed)
    device = get_device()
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(cfg["results_dir"]) / "logs" / "fewshot.log")
    save_config_snapshot(cfg, Path(cfg["results_dir"]) / "configs" / "fewshot.json")
    logger.info("Few-shot | backbones=%s pcts=%s folds=%s seed=%d | device=%s",
                backbones, pcts, folds, args.seed, device)

    rows = []
    for fold_id in folds:
        _, val_loader, test_loader = get_fold_loaders(
            fold_dir=cfg["fold_dir"], fold_id=fold_id, batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"], root_dir=cfg["root_dir"], image_root=cfg["image_root"],
            use_weighted_sampler=False, val_frac=cfg.get("val_frac", 0.2))
        monitor = val_loader if val_loader is not None else test_loader
        for pct in pcts:
            fs_loader, dropped = get_fewshot_loader(
                fold_dir=cfg["fold_dir"], n_shot=pct, fold_id=fold_id, batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"], root_dir=cfg["root_dir"], image_root=cfg["image_root"])
            if dropped:
                logger.warning("fewshot_%s fold%d: dropped %d rows colliding with test (anti-leakage).",
                               pct, fold_id, dropped)
            for backbone in backbones:
                rows.append(train_one(cfg, device, logger, backbone, pct, fold_id, args.seed,
                                      fs_loader, monitor, test_loader))

    res = pd.DataFrame(rows)
    res_path = Path(cfg["results_dir"]) / "fewshot_results.csv"
    if res_path.exists():
        res = pd.concat([pd.read_csv(res_path), res]).drop_duplicates(
            subset=["backbone", "data_pct", "fold", "seed"], keep="last")
    res.to_csv(res_path, index=False)

    # Summary mean±std across folds, canonical seed only (other seeds are robustness
    # checks living alongside in fewshot_results.csv — summarise_folds() would otherwise
    # conflate per-fold and per-seed variance if both are present for the same fold).
    canon = res[res["seed"] == SEED]
    summ_rows = []
    for (bb, pct), sub in canon.groupby(["backbone", "data_pct"]):
        s = summarise_folds(sub, METRIC_COLS)
        s.insert(0, "data_pct", pct); s.insert(0, "backbone", bb)
        summ_rows.append(s)
    summ = pd.concat(summ_rows, ignore_index=True)
    summ.to_csv(Path(cfg["results_dir"]) / "fewshot_summary.csv", index=False)

    # Data-efficiency curves: F1 & PR-AUC vs data%
    fig_dir = Path(cfg["figures_dir"]) / "fewshot"; fig_dir.mkdir(parents=True, exist_ok=True)
    for bb in backbones:
        sub = canon[canon["backbone"] == bb]
        if sub.empty:
            continue
        agg = sub.groupby("data_pct").agg(f1=("f1", "mean"), f1s=("f1", "std"),
                                          pr=("pr_auc", "mean"), prs=("pr_auc", "std")).reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(agg["data_pct"], agg["f1"], yerr=agg["f1s"].fillna(0), marker="o", label="F1")
        ax.errorbar(agg["data_pct"], agg["pr"], yerr=agg["prs"].fillna(0), marker="s", label="PR-AUC")
        ax.set_xlabel("Labeled data %"); ax.set_ylabel("Score"); ax.set_title(f"Data efficiency — {bb}")
        ax.legend()
        fig.tight_layout(); fig.savefig(fig_dir / f"{bb}_curve.png", dpi=150); plt.close(fig)

    print(summ.to_string(index=False))
    logger.info("Saved fewshot_results.csv + fewshot_summary.csv + curves")


if __name__ == "__main__":
    main()
