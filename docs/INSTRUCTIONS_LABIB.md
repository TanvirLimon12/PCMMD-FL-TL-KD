# INSTRUCTIONS — Labib

**Track:** Federated Learning (FedAvg + FedProx) · Client heterogeneity · Communication cost · Statistical validation
**Branch:** `labib-federated`
**Research questions:** RQ3 (FL vs centralized under non-IID), RQ3a (IID vs non-IID), RQ3b (FedProx stability), RQ3c (communication cost)

You must use the **same frozen** EDA outputs Abrar uses — no split generation in the FL code.
Each patient = one FL client. Seed = 42, image = 224×224 RGB. A patient-disjoint **validation**
set is held out of the federation to pick the best global model; the TEST fold is scored once
(leakage-safe — handled for you).

---

## 0. One-time setup

### Option A — Kaggle (GPU, recommended)

1. **New Notebook** → **Settings → Accelerator → GPU**.
2. **+ Add Data** → attach BOTH datasets:
   - EDA-outputs dataset (`pcmmd_eda_outputs/fold_1.csv … fold_5.csv`)
   - raw-image dataset (cropped cell PNGs)
3. **Clone + install:**
   ```python
   !git clone <YOUR_REPO_URL> /kaggle/working/PCMMD-repo
   %cd /kaggle/working/PCMMD-repo/PCMMD
   !pip install -q -r requirements.txt
   ```
4. **Auto-detect data paths and patch the configs (run once):**
   ```python
   import glob, pathlib, collections, yaml

   fold_hits = glob.glob('/kaggle/input/**/fold_1.csv', recursive=True)
   assert fold_hits, "fold_1.csv not found — attach the EDA-outputs dataset."
   fold_dir = str(pathlib.Path(fold_hits[0]).parent)

   exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
   counts = collections.Counter()
   for p in glob.glob('/kaggle/input/*'):
       counts[p] = sum(1 for f in glob.glob(p + '/**/*', recursive=True)
                       if pathlib.Path(f).suffix.lower() in exts)
   image_root = counts.most_common(1)[0][0]
   print("fold_dir =", fold_dir, "| image_root =", image_root, "(", counts[image_root], "imgs )")

   for cfg in glob.glob('configs/*.yaml'):
       d = yaml.safe_load(open(cfg))
       d['data_root'] = fold_dir; d['fold_dir'] = fold_dir; d['image_root'] = image_root
       yaml.safe_dump(d, open(cfg, 'w'), sort_keys=False)
   ```
5. **Verify clients + heterogeneity (Stage T1) before training:**
   ```python
   from data.folds import validate_fold, compute_client_stats
   print(validate_fold(fold_dir, 1))            # patient_leakage == [], n_val_patients > 0
   print(compute_client_stats(fold_dir, 1))     # 10 patients, plasma% high (mm) vs low (normal)
   ```

### Option B — Local VS Code
```bash
cd PCMMD
pip install -r requirements.txt
# edit configs/fedavg.yaml + configs/fedprox.yaml: set data_root (fold dir) and image_root (images)
```

---

## 1. What to run (in order)

```bash
# ── T2  Client heterogeneity — stats + figures (non-IID story) ───────────────
python analyze_clients.py --config configs/fedavg.yaml --fold 1
#   -> results/client_stats.csv
#      figures/clients/{plasma_percentage,label_skew_heatmap,quantity_skew,diagnosis_distribution}.png

# ── T4  FedAvg — IID sanity check, then patient non-IID ──────────────────────
python train_fedavg.py  --config configs/fedavg.yaml --distribution iid
python train_fedavg.py  --config configs/fedavg.yaml --distribution non-IID
#   -> results/fedavg_results.csv (best-round + final-round), fedavg_round_logs.csv,
#      communication_analysis.csv, client_analysis.csv,
#      figures/fedavg/{f1,loss,pr_auc}_curve_*.png, checkpoints/fedavg/*.pth

# ── T5  FedProx — mu sweep {0.001, 0.01, 0.1}, non-IID and IID ───────────────
python train_fedprox.py --config configs/fedprox.yaml --distribution non-IID
python train_fedprox.py --config configs/fedprox.yaml --distribution iid
#   compute-limited? run a single mu:  python train_fedprox.py --config configs/fedprox.yaml --mu 0.01
#   -> results/fedprox_results.csv, fedprox_round_logs.csv,
#      figures/fedprox/{f1,loss}_curve_*.png, checkpoints/fedprox/*.pth

# ── T9  Statistical validation (AFTER Abrar pushes centralized_results.csv) ──
python statistical_analysis.py --config configs/centralized.yaml
#   -> results/statistical_analysis.csv (mean±std, 95% CI),
#      results/statistical_pairwise.csv (Wilcoxon + paired-t + Cohen's d + interpretation):
#      centralized-vs-FedAvg, centralized-vs-FedProx, FedAvg-vs-FedProx
```

> To compare cleanly with Abrar: keep `backbone` the same across FedAvg and FedProx, and across
> IID/non-IID runs. Default is `mobilenet_v3` (fast); switch to `efficientnet_b0` in both FL
> configs if compute allows. All runs use the same `fold_id`, `num_rounds`, `local_epochs`.

### Config knobs you control (configs/fedavg.yaml, configs/fedprox.yaml)
- `distribution`: `non-IID` (one client per patient) | `iid` (also via `--distribution`).
- `num_rounds`, `local_epochs`, `batch_size`, `learning_rate`, `val_frac`.
- FedProx `mu_values: [0.001, 0.01, 0.1]` (or `--mu` for a single value).
- `fold_id` (run folds 1..5 by changing this to get fold-level paired stats).

---

## 2. Expected outputs (your deliverables)

```
results/fedavg_results.csv      results/fedprox_results.csv
results/fedavg_round_logs.csv   results/fedprox_round_logs.csv
results/client_stats.csv        results/client_analysis.csv
results/communication_analysis.csv
results/statistical_analysis.csv   results/statistical_pairwise.csv
figures/fedavg/  figures/fedprox/  figures/clients/
checkpoints/fedavg/  checkpoints/fedprox/   (*.pth — NOT pushed)
```

Paper mapping: Table 4 ← `fedavg_results.csv` + `fedprox_results.csv`; Table 6 ← `communication_analysis.csv`;
FL convergence figures; client heterogeneity + per-client figures; stats paragraph ← `statistical_pairwise.csv`.

---

## 3. Download + push to GitHub

```bash
git checkout -b labib-federated
git add results/ figures/ docs/experiment_log.md configs/
git commit -m "Labib: FedAvg + FedProx + heterogeneity + stats"
git push origin labib-federated
# open a Pull Request into 'dev'
```

**Commit:** code + result CSVs + figures + config snapshots.
**Never commit:** `*.pth` checkpoints, raw images, the dataset (already in `.gitignore`).

---

## 4. Acceptance checklist
- [ ] Same frozen EDA files as Abrar; no split generated inside FL code.
- [ ] All 10 patients verified as clients; `client_stats.csv` matches EDA counts.
- [ ] IID and non-IID results clearly separated; FedAvg vs FedProx use comparable settings.
- [ ] Best round AND final round reported; per-round logs saved.
- [ ] Communication table: rounds-to-best, model size, estimated comm MB, runtime.
- [ ] Statistical comparison uses fold-level paired results (run multiple folds for ≥2 pairs).
- [ ] Every figure has a matching CSV; best global checkpoints + config snapshots saved.
- [ ] Recorded each run in `docs/experiment_log.md`.
