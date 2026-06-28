# PROJECT STATUS — AI Radiomics GBM Pipeline
**Last updated: 2026-06-28**

---

## Current Goal
Extract PyRadiomics features from 25 downloaded NIfTI patients (tumor + peritumoral rings),
then train and compare ML models: **tumor-only vs tumor+peritumoral rings** — the core
scientific contribution ("beyond visible segmentation").

---

## What Has Been Implemented

### Environment
- Conda env: `cfb-gbm-radiomics` (Python 3.11, PyRadiomics 3.0.1 via conda-forge, numpy 1.x)
- Activate: `conda activate cfb-gbm-radiomics`
- All imports verified working: SimpleITK, pyradiomics, sklearn, shap, xgboost, lightgbm

### Data
- **261 patients** from pre-extracted TSV (TCIA CFB-GBM dataset)
- **25 patients** NIfTI downloaded: `data/raw/cfb_gbm/001/` through `025/`
- Image resolution: 512×512×208 voxels, spacing 0.5×0.5×1.0 mm (very high res)
- Peritumoral rings created for all 25 patients: `data/processed/regions/`
- QC figures verified correct: `results/figures/qc/` (25 × 4-panel PNG)

### Pipeline Scripts
| Script | Status | Purpose |
|--------|--------|---------|
| `src/build_ml_dataset_tsv.py` | ✅ Done | Builds ml_dataset.csv from pre-extracted TSV |
| `src/create_regions_nifti.py` | ✅ Done | Creates tumor_mask + ring_5mm + ring_10mm NIfTI |
| `src/extract_radiomics_nifti.py` | ❌ Blocked | Extracts features from NIfTI (see below) |
| `src/train_ml_models.py` | ✅ Done | Trains 6 models × 3 experiments with CV |
| `src/explain_models.py` | ✅ Done | SHAP beeswarm, bar, waterfall, permutation importance |
| `src/make_figures.py` | ✅ Done | ROC curves, confusion matrix, workflow diagram |

### ML Results (from 261-patient TSV dataset)
Trained on `data/features/ml_dataset.csv` — 261 patients, 139 features, OS label (median=50.5 weeks)

| Experiment | Best Model | AUC | 95% CI |
|------------|-----------|-----|--------|
| Volume_Only | SVM_RBF | 0.627 | 0.562–0.692 |
| Intratumoral_Only | LogReg_ElasticNet | **0.710** | 0.648–0.767 |
| All_Radiomics | RandomForest | 0.670 | 0.607–0.735 |

Best overall: `Intratumoral_Only__LogReg_ElasticNet` (AUC=0.710)

### Generated Figures
- `results/figures/roc_curves.png` — ROC curves for all models
- `results/figures/shap_summary.png` — SHAP beeswarm
- `results/figures/shap_bar.png` — Top features by SHAP
- `results/figures/shap_waterfall_p0.png` — Waterfall for patient 0
- `results/figures/permutation_importance.png`
- `results/figures/qc/*.png` — 25 patient QC figures

---

## Current Blocker: NIfTI Extraction Failing

**Problem:** `data/features/radiomics_25patients.csv` has only 2 columns (PatientID, Timepoint) — extraction produced 0 features.

**Root cause:** Images are 512×512×208 at 0.5mm spacing — extremely large. PyRadiomics hangs or times out per patient (~30+ min for 1 patient without resampling).

**Fix applied:** `config/pyradiomics_params.yaml` now has `resampledPixelSpacing: [2, 2, 2]` which should reduce processing to ~2-3 min per patient.

**Status:** Extraction has been attempted multiple times but keeps showing 0% progress for >1 hour. Need to re-run and confirm it actually completes for patient 001.

---

## Important Decisions
- OS binary label: median split at **50.5 weeks** survival
- Peritumoral rings: **0→5mm** (ring_5mm) and **5→10mm** (ring_10mm) from GTV surface
- Sequences used: T1Gd and FLAIR (primary diagnostic sequences for GBM)
- No LoG/Wavelet image types (disabled for speed)
- 5-fold stratified cross-validation, 1000-bootstrap 95% CI for AUC
- `resampledPixelSpacing: [2, 2, 2]` — PyRadiomics internal resampling (original files unchanged)

---

## Key Files
```
config/
  pyradiomics_params.yaml     ← params with 2mm resampling, no LoG/Wavelet
  config.yaml                 ← project paths and ML settings
data/
  features/
    ml_dataset.csv            ← 261 patients, 139 features (from TSV)
    radiomics_25patients.csv  ← 25 patients NIfTI extract (currently EMPTY)
  processed/
    regions/001-025/t0/       ← tumor_mask.nii.gz, ring_5mm.nii.gz, ring_10mm.nii.gz
  raw/cfb_gbm/001-025/t0/     ← *_t1gd.nii.gz, *_flair.nii.gz, *_gtv.nii.gz
results/
  tables/model_comparison.csv ← all model AUC results
  models/best_model.joblib    ← saved best model
  figures/                    ← all plots
```

---

## Commands That Work

```cmd
# Activate environment
conda activate cfb-gbm-radiomics
cd C:\Users\USER\Projekti\cfb-gbm-ai-radiomics

# Step 1: Build ML dataset from TSV (261 patients) — already done
python src/build_ml_dataset_tsv.py

# Step 2: Create peritumoral rings — already done
python src/create_regions_nifti.py --input data/raw/cfb_gbm --output data/processed/regions

# Step 3: Extract radiomics from NIfTI — BLOCKED, test with 1 patient first:
python src/extract_radiomics_nifti.py --input data/raw/cfb_gbm --regions data/processed/regions --output data/features/radiomics_25patients.csv --patients 001

# Step 4: Train ML models (works on TSV dataset)
python src/train_ml_models.py --input data/features/ml_dataset.csv --output results/

# Step 5: SHAP explainability
python src/explain_models.py --model results/models/best_model.joblib --input data/features/ml_dataset.csv
```

---

## Exact Next Steps

1. **Fix NIfTI extraction** — cancel any running process, then run single-patient test:
   ```cmd
   python src/extract_radiomics_nifti.py --input data/raw/cfb_gbm --regions data/processed/regions --output data/features/radiomics_25patients.csv --patients 001
   ```
   Should complete in ~3-5 min with 2mm resampling. If it hangs again, reduce feature classes (keep only firstorder + shape).

2. **Once extraction works** — run all 25 patients and build peritumoral ML dataset:
   ```cmd
   python src/extract_radiomics_nifti.py --input data/raw/cfb_gbm --regions data/processed/regions --output data/features/radiomics_25patients.csv
   ```

3. **Train peritumoral comparison** — new experiment comparing Ring5/Ring10 vs tumor-only on 25 patients.

4. **Fill in report** — `docs/seminar_report.md` results section needs actual numbers from above.

5. **Set up GitHub repository** — push code only (no patient data, no large NIfTI files).

---

## Known Issues / Watch Out For
- `radiomics_25patients.csv` — currently empty (only PatientID/Timepoint columns)
- 25 patients is too small for robust ML (use for peritumoral feature exploration/visualization only)
- Main ML results (AUC=0.710) are from 261-patient TSV dataset, not NIfTI
- PyRadiomics shows GLCM warning message — this is normal, not an error
- Do NOT include patient data or NIfTI files in GitHub repository
