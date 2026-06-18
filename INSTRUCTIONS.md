# INSTRUCTIONS — Team Split

> **Per-person step-by-step guides (with Kaggle setup):**
> [docs/INSTRUCTIONS_ABRAR.md](docs/INSTRUCTIONS_ABRAR.md) · [docs/INSTRUCTIONS_TANJID.md](docs/INSTRUCTIONS_TANJID.md)
> This file is the shared overview.

Two students, two branches, one shared `data/` + `models/` + `utils/` core.
Seed is fixed at **42**, image size **224×224 RGB**, folds are **frozen** — never
regenerate a split. All early stopping / model selection uses a patient-disjoint
**validation** set carved from train (see README → "Leakage safeguards"); the TEST
fold is scored once.

Sanity check before any run:

```bash
python -c "from data.folds import validate_fold; print(validate_fold('<fold_dir>', 1))"
# expect patient_leakage == []  AND  n_val_patients > 0
```

---

## Student 1 — Abrar  (branch: `abrar-centralized`)

Owns: **centralized baselines · few-shot/data-efficiency · knowledge distillation · deployment**.
RQ1 (centralized accuracy), RQ2 (data efficiency), RQ4 (lightweight student).

### Commands

```bash
# A2/A3 — centralized 5-fold CV for each required backbone
python train_centralized.py --config configs/centralized.yaml --backbone resnet50
python train_centralized.py --config configs/centralized.yaml --backbone efficientnet_b0
python train_centralized.py --config configs/centralized.yaml --backbone mobilenet_v3

# A4 — error analysis on the best model (add --panel to render example crops)
python error_analysis.py --config configs/centralized.yaml \
    --weights checkpoints/centralized/efficientnet_b0_fold1.pth --fold 1 --tag effnet --panel

# A5 — few-shot / data-efficiency (EfficientNet-B0 + MobileNetV3 on 5/10/20/50/100%)
python train_fewshot.py --config configs/fewshot.yaml

# A6/A7 — knowledge distillation (teacher = efficientnet_b0; T sweep) + deployment
python train_kd.py --config configs/kd.yaml
```

Loss is set in `configs/centralized.yaml` (`loss: weighted_ce | ce | focal`) and recorded in
every config snapshot. TL protocol via `finetune_mode: full | frozen | partial`.

### Expected outputs

```
results/centralized_results.csv      results/centralized_summary.csv
results/fewshot_results.csv          results/fewshot_summary.csv
results/kd_results.csv               results/kd_deployment.csv
results/error_*_misclassified.csv    results/curves/*_points.csv
checkpoints/centralized/  checkpoints/fewshot/  checkpoints/kd/
figures/centralized/  figures/fewshot/  figures/kd/  figures/errors/
```

---

## Student 2 — Tanjid  (branch: `tanjid-federated`)

Owns: **FedAvg · FedProx · client heterogeneity · communication · statistical validation**.
RQ3 (FL vs centralized under non-IID), RQ3a (IID vs non-IID), RQ3b (FedProx stability), RQ3c (comm cost).

### Commands

```bash
# T2 — client heterogeneity figures + per-patient stats
python analyze_clients.py --config configs/fedavg.yaml --fold 1

# T4 — FedAvg: IID sanity-check then patient non-IID
python train_fedavg.py  --config configs/fedavg.yaml --distribution iid
python train_fedavg.py  --config configs/fedavg.yaml --distribution non-IID

# T5 — FedProx: mu sweep {0.001, 0.01, 0.1}, IID and non-IID
python train_fedprox.py --config configs/fedprox.yaml --distribution non-IID
python train_fedprox.py --config configs/fedprox.yaml --distribution iid
# (compute-limited: python train_fedprox.py --config configs/fedprox.yaml --mu 0.01)

# T9 — statistical validation (run after Abrar shares centralized_results.csv)
python statistical_analysis.py --config configs/centralized.yaml
```

Backbone defaults to MobileNetV3 (speed); switch to `efficientnet_b0` in the config if compute allows.
Use all 10 clients per round (minus held-out val patients). Best round and final round are both reported.

### Expected outputs

```
results/fedavg_results.csv     results/fedprox_results.csv
results/fedavg_round_logs.csv  results/fedprox_round_logs.csv
results/client_stats.csv       results/client_analysis.csv
results/communication_analysis.csv
results/statistical_analysis.csv  results/statistical_pairwise.csv
checkpoints/fedavg/  checkpoints/fedprox/
figures/fedavg/  figures/fedprox/  figures/clients/
```

---

## Shared rules
- Never modify fold CSVs or regenerate splits. Each patient = one FL client.
- Smoke test first: centralized `--folds 1`; FL with a small `num_rounds`.
- Every CSV carries model/fold/seed; every plot has a matching CSV. No screenshot-only results.
- Commit code + CSV outputs; **never** commit `*.pth` or raw images (see `.gitignore`).
- Record each run in `docs/experiment_log.md`.
