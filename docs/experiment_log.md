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
