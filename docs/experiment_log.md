# Experiment Log

Record every run here. Keep it append-only. One row per run.

| Date | Student | Script | Config / overrides | Backbone | Folds/Rounds | Key result | Checkpoint | Notes |
|------|---------|--------|--------------------|----------|--------------|-----------|------------|-------|
| 2026-06-17 | — | (project scaffolded) | — | — | — | GitHub/Kaggle-ready pipeline built | — | folds frozen, seed 42 |
| 2026-06-18 | — | plasma-eda (run locally) | seed 42 | — | 5 folds | EDA outputs frozen → data/eda/ (2026 crops, 10 patients) | — | folds patient-disjoint, leak=[]; few-shot anti-leak verified |

## Template

```
| YYYY-MM-DD | Abrar/Tanjid | train_*.py | configs/X.yaml --flag | resnet50 | 5-fold / 50r | f1=0.__ acc=0.__ | checkpoints/.../*.pth | ... |
```

## Conventions
- Seed is always 42; do not change.
- Image size 224×224 RGB; folds are frozen — never regenerate.
- Positive class = plasma (idx 0): sensitivity = plasma recall, specificity = non_plasma recall.
- Paste the printed summary table into the Notes cell or attach the CSV path.

| 2026-06-22 | Tanjid | analyze_clients.py | configs/fedavg.yaml | — | 1 fold | Heterogeneity mapped | — | mm: 35-41%, normal: 4-9% |
| 2026-06-22 | Tanjid | train_fedavg.py | configs/fedavg.yaml --dist iid | mobilenet_v3 | 1 fold / 50r | Completed | — | Device mismatch patched |
| 2026-06-22 | Tanjid | train_fedavg.py | configs/fedavg.yaml --dist non-IID | mobilenet_v3 | 1 fold / 50r | Completed | — | — |
| 2026-06-22 | Tanjid | train_fedprox.py | configs/fedprox.yaml --dist non-IID | mobilenet_v3 | 1 fold / 50r | Completed | — | mu=[0.001, 0.01, 0.1] sweep |
| 2026-06-22 | Tanjid | train_fedprox.py | configs/fedprox.yaml --dist iid | mobilenet_v3 | 1 fold / 50r | Completed | — | — |
| 2026-06-22 | Tanjid | statistical_analysis.py | configs/centralized.yaml | — | — | Stats generated | — | Compared vs Abrar's centralized results |
