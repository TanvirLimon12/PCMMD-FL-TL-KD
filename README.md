# PCMMD — Federated Learning, Transfer Learning & Knowledge Distillation

Plasma Cell / Multiple Myeloma classification from bone-marrow cell crops.
Binary task: **plasma** vs **non_plasma** (cell-level label).

This repo benchmarks several learning regimes on the **frozen** PCMMD folds:

| Regime | Scripts | Idea |
|---|---|---|
| **Centralized** (upper bound) | `train_centralized.py` | All data in one place, 5-fold CV |
| **Few-shot / data-efficiency** | `train_fewshot.py` | Train on 5/10/20/50/100 % frozen pools |
| **Knowledge Distillation** | `train_kd.py` | EfficientNet-B0 teacher → MobileNetV3 student (edge deployment) |
| **Federated** | `train_fedavg.py`, `train_fedprox.py` | One client per patient (non-IID) or IID shards |
| **Analysis** | `evaluate.py`, `error_analysis.py`, `analyze_clients.py`, `statistical_analysis.py` | Metrics, error panels, heterogeneity, paired stats |

> **Data is never regenerated.** Folds, few-shot subsets and metadata come from the
> EDA stage (`plasma-eda.ipynb`). Seed = **42**, image size = **224×224 RGB**, everywhere.

### Leakage safeguards (read this)
- **Target** is the cell-level `label` (plasma/non_plasma). `patient_diagnosis` (mm/normal)
  is **blocked** as a target — it is constant per patient and leaks identity.
- **Patient-disjoint folds**: `role` train/test never share a patient (checked by `validate_fold`).
- **No test-set model selection**: a patient-disjoint **validation** set is carved from the
  train patients (`val_frac`, stratified by diagnosis, seed 42). Early stopping & best-checkpoint
  selection use VAL only; the TEST fold is scored exactly once. FL holds the val patients out of
  the federation entirely.
- **Few-shot anti-leakage**: few-shot pool rows whose image identity (md5, else basename) collide
  with the test fold are dropped before training.
- **Privacy framing**: FL here is *decentralized training simulation* (only model updates are
  aggregated, raw images stay local). It does **not** implement differential privacy — do not
  claim formal DP unless DP-SGD/secure-aggregation is added.

---

## Folder structure

```
PCMMD/
├── train_centralized.py     # 5-fold CV, AdamW, val-based early stop, full metrics+ECE+figures
├── train_fewshot.py         # data-efficiency curves on frozen 5/10/20/50/100% pools
├── train_kd.py              # baseline vs KD student (T sweep) + deployment analysis
├── train_fedavg.py          # FedAvg — IID & patient non-IID, best/final round
├── train_fedprox.py         # FedProx — mu sweep {0.001,0.01,0.1}, IID & non-IID
├── evaluate.py              # checkpoint → metrics, predictions, ROC/PR/reliability, per-patient
├── error_analysis.py        # misclassified CSV + TP/TN/FP/FN panel (plasma FN focus)
├── analyze_clients.py       # client_stats.csv + heterogeneity figures
├── statistical_analysis.py  # mean±std, 95% CI, Wilcoxon/paired-t, bootstrap, interpretation
│
├── models/                  # efficientnet.py · mobilenet.py · resnet.py + factory (freeze modes)
├── data/                    # dataset.py · transforms.py · folds.py (val split, anti-leakage)
├── fl/                      # fedavg.py · fedprox.py · client.py · engine.py
├── utils/                   # common.py · metrics.py (ECE/bootstrap) · plots.py · losses.py
├── configs/                 # centralized · fewshot · kd · fedavg · fedprox  (YAML)
│
├── results/                 # CSV outputs + curves/ + logs/ + configs/ (snapshots)
├── checkpoints/             # <task>/<backbone>_*.pth
├── figures/                 # PNG figures per task
├── docs/experiment_log.md
├── README.md · INSTRUCTIONS.md · requirements.txt · .gitignore
```

---

## Data layout & image resolution (important)

The fold CSVs (`fold_1.csv … fold_5.csv`) carry these columns:

```
path, patient_id, patient_diagnosis, label, image_type, set_name, fold, role, md5
```

- **Target** = `label` ∈ {plasma, non_plasma}. **Never** `patient_diagnosis` (patient-level
  mm/normal — it leaks patient identity; guarded in `data/dataset.py`).
- **Split** = `role` ∈ {train, test}. Splits are **patient-disjoint** → no leakage
  (validated by `data.folds.validate_fold`).
- `path` is an **absolute EDA-machine path** that does not exist on Kaggle. The dataset
  therefore resolves images by **basename** against an index built from `image_root`.
  → Always set `image_root` to the folder containing the raw images.

Two inputs are needed:
1. **EDA outputs** (`fold_*.csv`, `fewshot_*.csv`, `patient_cells_metadata.csv`) → `fold_dir` / `data_root`
2. **Raw images** → `image_root`

### Bundled data (already in this repo)
The EDA stage has **already been run** (seed 42) and its small outputs are committed at
**`data/eda/`**: `fold_1.csv … fold_5.csv`, `fewshot_5/10/20/50/100.csv`, `metadata.csv`,
`client_stats_eda.csv`. Configs default to this folder, so **local runs work out of the box**.

The 2026 cropped cell images (~25 MB) live at **`data/patient_cells/`** and are **git-ignored**
(images never go in git). For a fresh clone you must supply them:
- **Local:** keep/restore `data/patient_cells/` (the crops the fold CSV `path`/basename refer to).
- **Kaggle:** zip `data/patient_cells/` → upload as a Kaggle dataset → attach it, then run the
  path-patch cell in `docs/INSTRUCTIONS_ABRAR.docx` / `INSTRUCTIONS_LABIB.docx` to set `image_root`.

> Do **not** re-run the EDA notebook or regenerate folds — they are frozen and patient-disjoint
> (verified: all 5 folds have `patient_leakage == []`).

---

## Installation

```bash
pip install -r requirements.txt
```

`scipy` and `thop` are optional — the pipeline degrades gracefully (NaN p-values /
skipped FLOPs) if they are absent.

---

## Local execution (VS Code)

Edit a config so the paths point at your local copies:

```yaml
# configs/centralized.yaml
data_root:  "/path/to/pcmmd_eda_outputs"   # holds fold_*.csv
image_root: "/path/to/raw_images"          # holds the cell crops
```

Then:

```bash
cd PCMMD
python train_centralized.py --config configs/centralized.yaml --folds 1   # quick smoke test
python train_centralized.py --config configs/centralized.yaml             # full 5-fold CV
```

CUDA is used automatically when available, otherwise CPU.

---

## Kaggle execution

1. New Kaggle Notebook → **Add Data**: attach the EDA-outputs dataset and the raw-image dataset.
2. Check the real mount paths (they vary by dataset slug):
   ```python
   !ls /kaggle/input
   !ls /kaggle/input/<your-eda-dataset>/pcmmd_eda_outputs | head
   ```
3. Clone this repo and install:
   ```bash
   !git clone <repo-url> && cd PCMMD && pip install -r requirements.txt
   ```
4. Point the config at the mounts (override inline or edit the YAML):
   ```python
   # e.g. data_root=/kaggle/input/new-eda-dataset/pcmmd_eda_outputs
   #      image_root=/kaggle/input/plasma-dataset
   ```
5. Run your assigned command (see `INSTRUCTIONS.md`).
6. Download `results/`, `figures/`, `checkpoints/`. Push **only code + CSV outputs** back to GitHub
   (never raw images or `*.pth`).

---

## Outputs

| File | Produced by |
|---|---|
| `results/centralized_results.csv` / `centralized_summary.csv` | `train_centralized.py` |
| `results/fewshot_results.csv` / `fewshot_summary.csv` | `train_fewshot.py` |
| `results/kd_results.csv` / `kd_deployment.csv` | `train_kd.py` |
| `results/fedavg_results.csv` / `fedavg_round_logs.csv` | `train_fedavg.py` |
| `results/fedprox_results.csv` / `fedprox_round_logs.csv` | `train_fedprox.py` |
| `results/client_stats.csv` / `client_analysis.csv` | FL scripts / `analyze_clients.py` |
| `results/communication_analysis.csv` | FL scripts |
| `results/error_*_misclassified.csv` | `error_analysis.py` |
| `results/statistical_analysis.csv` / `statistical_pairwise.csv` | `statistical_analysis.py` |
| `results/curves/*_points.csv` | ROC/PR raw points (centralized, evaluate) |
| `figures/<task>/*.png` | every script (confusion/ROC/PR/reliability/curves/heatmaps) |
| `checkpoints/<task>/*.pth` | every training script |

---

## Experiment workflow

```
centralized (resnet50 / efficientnet_b0 / mobilenet_v3, 5-fold)
        │
        ├── teacher = efficientnet_b0 ── train_kd.py ── MobileNetV3 student (+deployment)
        │
        └── upper-bound reference for ──► FedAvg / FedProx (IID & non-IID)
                                                │
                          analyze_clients.py ◄──┤ (heterogeneity)
                        statistical_analysis.py ◄┘ (mean±std, CI, paired tests)
```

---

## Git branch workflow

```
main            # protected — no direct commits, release-only
 └── dev        # integration branch; PRs target here
      ├── abrar-centralized   # Student 1: centralized + few-shot + KD + deployment
      └── tanjid-federated    # Student 2: FedAvg + FedProx + client/comm/statistics
```

- No direct commits to `main`. Work in your assigned branch.
- Open a PR into `dev`; review before merge. `dev` → `main` only for releases.

See **INSTRUCTIONS.md** for per-student commands and expected outputs.
