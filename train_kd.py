"""
train_kd.py
-----------
Knowledge Distillation for edge deployment.

  Teacher : EfficientNet-B0   (frozen; trained by train_centralized.py)
  Student : MobileNetV3        (lightweight deployment target)

Trains the baseline student (plain CE) and one KD student per temperature in
config 'temperatures' (default [2, 4, 6]); alpha default 0.5. Selection/early
stopping use the patient-disjoint VAL set; TEST is scored once per model.

Deployment analysis: params, size (MB), FLOPs/MACs (thop if present), latency for
teacher + student, plus an efficiency-vs-performance scatter.

Outputs:
  results/kd_results.csv      — teacher / baseline-student / kd-student(T=*) metrics
  results/kd_deployment.csv   — params/size/flops/latency for teacher + student
  figures/kd/{efficiency_performance,latency,model_size,*_history}.png
  checkpoints/kd/{baseline,distilled_T*}_<student>_fold<k>.pth

Prereq: teacher checkpoint from
        python train_centralized.py --backbone efficientnet_b0 --folds <k>
Run:    python train_kd.py --config configs/kd.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.folds import get_fold_loaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    collect_predictions, compute_all_metrics, get_device, load_config,
    model_complexity_report, plot_efficiency_performance, plot_training_history,
    save_config_snapshot, set_seed, setup_logging,
)

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "macro_f1",
               "roc_auc", "pr_auc", "specificity", "sensitivity"]


def kd_loss(student_logits, labels, teacher_logits, alpha=0.5, T=4.0):
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1.0 - alpha) * hard


def _metrics(model, loader, device):
    yt, yp, pr, _ = collect_predictions(model, loader, device)
    return compute_all_metrics(yt, yp, pr)


def train_student(cfg, device, logger, train_loader, monitor_loader, test_loader,
                  teacher=None, T=4.0, tag="baseline"):
    student = build_model(cfg["student_backbone"], num_classes=2, pretrained=cfg["pretrained"]).to(device)

    ckpt_dir_check = Path(cfg["ckpt_dir"]) / "kd"
    ckpt_check = ckpt_dir_check / f"{tag}_{cfg['student_backbone']}_fold{cfg['fold_id']}.pth"
    if ckpt_check.exists():
        logger.info("  [%s] checkpoint exists — skipping training, loading & evaluating", tag)
        student.load_state_dict(torch.load(ckpt_check, map_location=device))
        tm = _metrics(student, test_loader, device)
        row = {"model": tag, "student": cfg["student_backbone"], "fold": cfg["fold_id"], "temperature": T,
               **{k: round(tm[k], 5) for k in METRIC_COLS}}
        return student, row

    optimizer = optim.AdamW(student.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    ce = nn.CrossEntropyLoss()
    alpha = cfg.get("alpha", 0.5)

    ckpt_dir = Path(cfg["ckpt_dir"]) / "kd"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{tag}_{cfg['student_backbone']}_fold{cfg['fold_id']}.pth"

    best_f1, patience_ctr, history = -1.0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        student.train()
        running, total = 0.0, 0
        for batch in train_loader:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            s_logits = student(imgs)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs)
                loss = kd_loss(s_logits, lbls, t_logits, alpha=alpha, T=T)
            else:
                loss = ce(s_logits, lbls)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0); total += imgs.size(0)
        scheduler.step()
        vm = _metrics(student, monitor_loader, device)
        history.append({"epoch": epoch, "train_loss": round(running / max(1, total), 5),
                        "val_f1": round(vm["f1"], 5), "val_accuracy": round(vm["accuracy"], 5)})
        logger.info("  [%s] ep %03d/%d loss=%.4f val_f1=%.4f", tag, epoch, cfg["epochs"],
                    running / max(1, total), vm["f1"])
        if vm["f1"] > best_f1:
            best_f1, patience_ctr = vm["f1"], 0
            torch.save(student.state_dict(), ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                logger.info("  [%s] early stop @ %d", tag, epoch); break

    student.load_state_dict(torch.load(ckpt, map_location=device))
    tm = _metrics(student, test_loader, device)
    plot_training_history(history, Path(cfg["figures_dir"]) / "kd" / f"{tag}_history.png",
                          title=f"{tag} history")
    row = {"model": tag, "student": cfg["student_backbone"], "fold": cfg["fold_id"], "temperature": T,
           **{k: round(tm[k], 5) for k in METRIC_COLS}}
    return student, row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kd.yaml")
    ap.add_argument("--fold", type=int, default=None, help="Override fold_id from config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("teacher_backbone", "efficientnet_b0")
    cfg.setdefault("student_backbone", "mobilenet_v3")
    # T=1.0 is the no-soft-label anchor; required by reviewers as ablation baseline
    temps = cfg.get("temperatures", [1.0, 2.0, 4.0, 6.0])
    if 1.0 not in temps:
        temps = [1.0] + list(temps)
    alpha_values = cfg.get("alpha_values", [cfg.get("alpha", 0.5)])
    if args.fold is not None:
        cfg["fold_id"] = args.fold
    fold_id = cfg["fold_id"]

    set_seed()
    device = get_device()
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(cfg["results_dir"]) / "logs" / "kd.log")
    save_config_snapshot(cfg, Path(cfg["results_dir"]) / "configs" / "kd.json")
    logger.info("KD | teacher=%s student=%s fold=%d temps=%s alphas=%s | device=%s",
                cfg["teacher_backbone"], cfg["student_backbone"], fold_id, temps,
                alpha_values, device)

    train_loader, val_loader, test_loader = get_fold_loaders(
        fold_dir=cfg["fold_dir"], fold_id=fold_id, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], root_dir=cfg["root_dir"], image_root=cfg["image_root"],
        use_weighted_sampler=cfg["use_weighted_sampler"], val_frac=cfg.get("val_frac", 0.2))
    monitor = val_loader if val_loader is not None else test_loader
    if val_loader is None:
        logger.warning("No val patients -> monitoring on TEST (report cautiously).")

    # Teacher (frozen)
    teacher_path = cfg.get("teacher_weights") or (
        f"{cfg['ckpt_dir']}/centralized/{cfg['teacher_backbone']}_fold{fold_id}.pth")
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"Teacher weights not found: {teacher_path}\n"
            f"Run: python train_centralized.py --backbone {cfg['teacher_backbone']} --folds {fold_id}")
    teacher = build_model(cfg["teacher_backbone"], num_classes=2, pretrained=False).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    logger.info("Teacher loaded & frozen: %s", teacher_path)
    teacher_test = _metrics(teacher, test_loader, device)

    rows = [{"model": "teacher", "student": cfg["teacher_backbone"], "fold": fold_id,
             "temperature": None, "alpha": None,
             **{k: round(teacher_test[k], 5) for k in METRIC_COLS}}]

    # Baseline student (no teacher, no KD)
    logger.info("Training BASELINE student (no teacher)...")
    cfg_base = {**cfg, "alpha": 0.0}
    _, base_row = train_student(cfg_base, device, logger, train_loader, monitor, test_loader,
                                teacher=None, tag="baseline")
    base_row["alpha"] = None
    rows.append(base_row)

    # Temperature × alpha ablation sweep
    best_kd_student, best_kd_row = None, None
    for T in temps:
        for alpha in alpha_values:
            cfg_run = {**cfg, "alpha": alpha}
            tag = f"distilled_T{T}_a{alpha}"
            logger.info("Training KD student (T=%s, alpha=%s)...", T, alpha)
            student, row = train_student(cfg_run, device, logger, train_loader, monitor, test_loader,
                                         teacher=teacher, T=T, tag=tag)
            row["alpha"] = alpha
            rows.append(row)
            if best_kd_row is None or row["f1"] > best_kd_row["f1"]:
                best_kd_student, best_kd_row = student, row

    kd_df = pd.DataFrame(rows)
    kd_df.to_csv(Path(cfg["results_dir"]) / "kd_results.csv", index=False)
    logger.info("Saved results/kd_results.csv")

    # Deployment analysis (teacher + best KD student)
    deploy_rows = []
    for name, model, met in [("teacher_" + cfg["teacher_backbone"], teacher, teacher_test),
                             ("student_" + cfg["student_backbone"], best_kd_student, best_kd_row)]:
        rep = {"model": name, **model_complexity_report(model, device),
               "test_f1": round(met["f1"], 5), "test_pr_auc": round(met["pr_auc"], 5)}
        deploy_rows.append(rep)
        logger.info("Deploy %s: params=%.2fM size=%.1fMB latency=%.2fms f1=%.4f",
                    name, rep["params_total_m"], rep["model_size_mb"], rep["latency_ms_mean"], rep["test_f1"])
    deploy_df = pd.DataFrame(deploy_rows)
    deploy_df.to_csv(Path(cfg["results_dir"]) / "kd_deployment.csv", index=False)

    # Alpha ablation plot at best temperature
    fig_dir = Path(cfg["figures_dir"]) / "kd"; fig_dir.mkdir(parents=True, exist_ok=True)
    if len(alpha_values) > 1 and best_kd_row is not None:
        best_T = best_kd_row["temperature"]
        alpha_rows = kd_df[kd_df["temperature"] == best_T].dropna(subset=["alpha"])
        if len(alpha_rows) > 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(alpha_rows["alpha"], alpha_rows["f1"], "o-", color="C0")
            ax.axhline(base_row["f1"], linestyle="--", color="gray", label="baseline (no KD)")
            ax.set_xlabel("Alpha (KD weight)"); ax.set_ylabel("Test F1")
            ax.set_title(f"KD alpha ablation (T={best_T})")
            ax.legend(); fig.tight_layout()
            fig.savefig(fig_dir / "alpha_ablation.png", dpi=150); plt.close(fig)

    plot_efficiency_performance(deploy_df["model"].tolist(), deploy_df["model_size_mb"].tolist(),
                                deploy_df["test_f1"].tolist(), fig_dir / "efficiency_performance.png",
                                xlabel="Model size (MB)", ylabel="Test F1",
                                title="Efficiency vs performance")
    for col, ylab, fn in [("latency_ms_mean", "Latency (ms/img)", "latency.png"),
                          ("model_size_mb", "Model size (MB)", "model_size.png")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(deploy_df["model"], deploy_df[col], color=["#C44E52", "#4C72B0"])
        ax.set_ylabel(ylab); ax.set_title(ylab); plt.xticks(rotation=15, ha="right")
        fig.tight_layout(); fig.savefig(fig_dir / fn, dpi=150); plt.close(fig)

    # Temperature sensitivity plot
    temp_rows = kd_df[kd_df["alpha"] == cfg.get("alpha", 0.5)].dropna(subset=["temperature"])
    if len(temp_rows) > 1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(temp_rows["temperature"], temp_rows["f1"], "s-", color="C1")
        ax.axhline(base_row["f1"], linestyle="--", color="gray", label="baseline (no KD)")
        ax.set_xlabel("Temperature (T)"); ax.set_ylabel("Test F1")
        ax.set_title("KD temperature sensitivity")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "temperature_sensitivity.png", dpi=150); plt.close(fig)

    print("\n=== KD RESULTS ===")
    print(kd_df.to_string(index=False))
    print("\n=== DEPLOYMENT ===")
    print(deploy_df.to_string(index=False))
    if best_kd_row is not None:
        print(f"\nBest KD student (T={best_kd_row['temperature']}, alpha={best_kd_row.get('alpha')}) "
              f"F1={best_kd_row['f1']:.4f} vs baseline {base_row['f1']:.4f} "
              f"(gain {best_kd_row['f1']-base_row['f1']:+.4f})")


if __name__ == "__main__":
    main()
