# INSTRUCTIONS — Abrar

**Track:** Centralized baselines · Few-shot / data-efficiency · Knowledge Distillation · Deployment · Error analysis
**Branch:** `abrar-centralized`
**Research questions:** RQ1 (centralized accuracy), RQ2 (data efficiency), RQ4 (lightweight student)

You must use the **frozen** EDA outputs (folds, few-shot pools). Never regenerate a split.
Seed = 42, image = 224×224 RGB. Model selection uses a patient-disjoint **validation** set
carved from train; the TEST fold is scored once (leakage-safe — handled for you).

---

## 0. One-time setup

### Option A — Kaggle (GPU, recommended)

1. **New Notebook** → right panel **Settings → Accelerator → GPU** (T4/P100).
2. **+ Add Data** → attach the **PCMMD dataset** (uploaded from `PCMMD_kaggle_dataset.zip`).
   It is self-contained: `pcmmd_eda_outputs/` (folds + few-shot + metadata) **and**
   `patient_cells/` (the cropped cell images). One dataset is enough.
3. **Clone the repo** (first code cell):
   ```python
   !git clone https://github.com/TanvirLimon12/PCMMD-FL-TL-KD /kaggle/working/PCMMD-repo
   %cd /kaggle/working/PCMMD-repo/PCMMD
   !pip install -q -r requirements.txt
   ```
   > The folds/few-shot CSVs are also bundled in the repo at `data/eda/` (so local runs work
   > with no setup). On Kaggle you still attach the dataset above for the **images**, then run
   > the patch cell below to point `image_root` at the mounted images.
4. **Point the configs at the mounted data.** Mount paths vary by dataset slug, so auto-detect them
   and patch every config (run this cell once):
   ```python
   import glob, pathlib, collections, yaml, os

   # locate the folds dir (the folder that holds fold_1.csv)
   fold_hits = glob.glob('/kaggle/input/**/fold_1.csv', recursive=True)
   assert fold_hits, "fold_1.csv not found — did you attach the PCMMD dataset?"
   fold_dir = str(pathlib.Path(fold_hits[0]).parent)

   # locate the image root = input subtree with the most image files
   exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
   counts = collections.Counter()
   for p in glob.glob('/kaggle/input/*'):
       n = sum(1 for f in glob.glob(p + '/**/*', recursive=True)
               if pathlib.Path(f).suffix.lower() in exts)
       counts[p] = n
   image_root = counts.most_common(1)[0][0]

   print("fold_dir   =", fold_dir)
   print("image_root =", image_root, "(", counts[image_root], "images )")

   for cfg in glob.glob('configs/*.yaml'):
       d = yaml.safe_load(open(cfg))
       d['data_root'] = fold_dir; d['fold_dir'] = fold_dir; d['image_root'] = image_root
       yaml.safe_dump(d, open(cfg, 'w'), sort_keys=False)
   print("patched:", glob.glob('configs/*.yaml'))
   ```
5. **Sanity check (no leakage):**
   ```python
   from data.folds import validate_fold
   print(validate_fold(fold_dir, 1))   # expect patient_leakage == []  AND  n_val_patients > 0
   ```

### Option B — Local VS Code (CPU/GPU)
```bash
cd PCMMD
pip install -r requirements.txt
# edit configs/*.yaml: set data_root (folder with fold_*.csv) and image_root (folder with images)
```

---

## 1. What to run (in order)

> Tip: on the first pass add `--folds 1` to centralized for a fast smoke test, then drop it for full 5-fold.

```bash
# ── A2/A3  Centralized 5-fold CV — run all three required backbones ──────────
python train_centralized.py --config configs/centralized.yaml --backbone resnet50
python train_centralized.py --config configs/centralized.yaml --backbone efficientnet_b0
python train_centralized.py --config configs/centralized.yaml --backbone mobilenet_v3
#   -> results/centralized_results.csv (per fold), centralized_summary.csv (mean±std),
#      figures/centralized/*, checkpoints/centralized/<backbone>_fold<k>.pth

# ── A4  Error analysis on the best model (FP/FN review; --panel renders crops) ─
python error_analysis.py --config configs/centralized.yaml \
    --weights checkpoints/centralized/efficientnet_b0_fold1.pth --fold 1 --tag effnet --panel
#   -> results/error_effnet_fold1_misclassified.csv, figures/errors/*

# ── A5  Few-shot / data-efficiency (EfficientNet-B0 + MobileNetV3, 5..100%) ───
python train_fewshot.py --config configs/fewshot.yaml
#   -> results/fewshot_results.csv, fewshot_summary.csv, figures/fewshot/*_curve.png

# ── A6/A7  Knowledge distillation + deployment ───────────────────────────────
#   PREREQ: the efficientnet_b0 teacher checkpoint for fold_id (default 1) must
#   exist — it is produced by the centralized run above.
python train_kd.py --config configs/kd.yaml
#   -> results/kd_results.csv (teacher/baseline/KD per T), kd_deployment.csv
#      (params/size/FLOPs/latency), figures/kd/efficiency_performance.png

# ── (optional) detailed single-checkpoint evaluation with curves + per-patient ─
python evaluate.py --config configs/centralized.yaml \
    --weights checkpoints/centralized/resnet50_fold1.pth --fold 1 --tag resnet50
```

### Config knobs you control (configs/centralized.yaml, configs/fewshot.yaml, configs/kd.yaml)
- `loss`: `weighted_ce` (default) | `ce` | `focal` — report which you used (saved in the snapshot).
- `finetune_mode`: `full` (default) | `frozen` | `partial` — transfer-learning protocol.
- `epochs`, `patience` (early stop on val F1), `batch_size`, `learning_rate`, `val_frac`.
- KD: `temperatures: [2,4,6]`, `alpha: 0.5`.

---

## 2. Expected outputs (your deliverables)

```
results/centralized_results.csv      results/centralized_summary.csv
results/fewshot_results.csv          results/fewshot_summary.csv
results/kd_results.csv               results/kd_deployment.csv
results/error_*_misclassified.csv    results/curves/*_points.csv
figures/centralized/  figures/fewshot/  figures/kd/  figures/errors/
checkpoints/centralized/  checkpoints/fewshot/  checkpoints/kd/   (*.pth — NOT pushed)
```

Paper mapping: Table 2 ← `centralized_summary.csv`; Table 3 ← `fewshot_summary.csv` + curve;
Table 5 ← `kd_results.csv` + `kd_deployment.csv`; ROC/PR + few-shot + error figures.

---

## 3. Download + push to GitHub

On Kaggle, **Save Version (Run All)** then download `results/` and `figures/`, OR commit from the notebook:

```bash
git checkout -b abrar-centralized        # work only on your branch
git add results/ figures/ docs/experiment_log.md configs/
git commit -m "Abrar: centralized + few-shot + KD results"
git push origin abrar-centralized
# open a Pull Request into 'dev'
```

**Commit:** code + result CSVs + figures + config snapshots.
**Never commit:** `*.pth` checkpoints, raw images, the dataset (already in `.gitignore`).

---

## 4. Acceptance checklist
- [ ] All runs used the frozen folds (`validate_fold` showed `patient_leakage == []`).
- [ ] `centralized_summary.csv` has mean ± std across 5 folds for all 3 backbones.
- [ ] Few-shot curve covers 5/10/20/50/100 % with the same test fold each time.
- [ ] KD table compares teacher vs normal student vs KD student; deployment metrics filled.
- [ ] Every figure has a matching CSV; best checkpoints + config snapshots saved.
- [ ] Recorded each run in `docs/experiment_log.md`.
