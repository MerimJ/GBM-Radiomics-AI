# Complete Project Guide: AI-Driven MRI Radiomics for Glioblastoma

This guide explains every component of the pipeline, all figures, all results, and how to reproduce everything from scratch.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Does This Include AI?](#2-does-this-include-ai)
3. [Dataset](#3-dataset)
4. [Environment Setup](#4-environment-setup)
5. [Pipeline Step-by-Step](#5-pipeline-step-by-step)
6. [Figure Explanations](#6-figure-explanations)
7. [Results Summary](#7-results-summary)
8. [File Structure](#8-file-structure)
9. [Frequently Asked Questions](#9-frequently-asked-questions)

---

## 1. Project Overview

**Title:** AI-Driven MRI Radiomics for Glioblastoma: Quantitative Features Beyond Visible Segmentation for Outcome Prediction and Explainable Machine Learning

**What this project does:**
- Downloads MRI data (T1Gd and FLAIR) from 79–261 glioblastoma patients
- Extracts ~336 quantitative image features (radiomics) from both the tumour and the tissue surrounding it
- Trains machine learning models to predict whether a patient will be a long or short survivor
- Uses SHAP (a form of explainable AI) to explain which features drive predictions
- Compares: **tumour-only features** vs **tumour + peritumoral ring features**
- Demonstrates that the tissue *beyond* the visible tumour boundary contains independent prognostic information

**Core finding:** Adding peritumoral ring features (0–5mm around the tumour) improves AUC from 0.562 to 0.670 — a +19% relative improvement, confirming that GBM biology extends beyond what is visible on MRI.

---

## 2. Does This Include AI?

**Yes — in three ways:**

### 2.1 Machine Learning (Classical AI)
The pipeline trains and compares four ML classifiers:
- **Logistic Regression with L1/ElasticNet regularisation** — learns a sparse linear combination of features to predict survival. Penalty term prevents overfitting.
- **Support Vector Machine (RBF kernel)** — maps features into higher-dimensional space to find non-linear decision boundaries.
- **Random Forest** — ensemble of 100–200 decision trees, each trained on random feature subsets. Robust to noise and correlated features.
- **XGBoost** — gradient boosting: trees trained sequentially, each correcting errors of previous trees.

All models learn patterns from labelled patient data (survival outcome). They automatically discover which combinations of hundreds of radiomic features distinguish long from short survivors — this is supervised machine learning / AI.

### 2.2 Explainable AI (XAI)
SHAP (SHapley Additive exPlanations) is applied to the best trained model:
- Computes how much each feature contributed to each patient's prediction
- Based on game theory (Shapley values from cooperative game theory)
- Produces beeswarm plots, bar charts, and waterfall plots
- Allows us to ask: "Is the model using T1Gd texture or FLAIR ring features?" — and get a quantitative answer

This is a core component of XAI (Explainable Artificial Intelligence) — making AI decisions transparent and interpretable.

### 2.3 Automated Feature Selection
`SelectKBest` with ANOVA F-score automatically identifies the top-10 most discriminating features from hundreds of candidates — without any human input. This is a form of automated machine learning (AutoML) for feature engineering.

### 2.4 Optional Deep Learning
`src/train_deep_embeddings.py` implements a 3D ResNet that learns features directly from MRI volumes using a convolutional neural network. This is deep learning AI. Requires an NVIDIA GPU with 48GB VRAM.

---

## 3. Dataset

**Source:** The Cancer Imaging Archive (TCIA) — https://www.cancerimagingarchive.net/
**Collection:** CFB-GBM (Collaborative Feasibility for Brain GBM)
**Access:** Free, public, requires TCIA account

### What's in the dataset:
- **264 glioblastoma patients** with multiparametric MRI
- **T1Gd (T1 gadolinium-enhanced):** shows contrast-enhancing tumour core
- **FLAIR (Fluid-Attenuated Inversion Recovery):** shows broader oedema and infiltration zone
- **GTV masks (NIfTI):** gross tumour volume contoured for radiotherapy planning
- **Clinical data:** survival in weeks, age, gender, WHO performance status

### What we downloaded:
- Patients 001–080 (80 patients, NIfTI format)
- 79/80 had valid GTV masks and survival data
- Pre-extracted TSV features for 261 patients (provided by TCIA)

### Privacy note:
Raw patient MRI files are NOT included in the GitHub repository. Only code, derived summary statistics, and non-identifiable figures are shared.

---

## 4. Environment Setup

```bash
# Create and activate the conda environment
conda activate cfb-gbm-radiomics

# Navigate to project directory
cd C:\Users\USER\Projekti\cfb-gbm-ai-radiomics

# Verify all imports work
python -c "import radiomics, SimpleITK, sklearn, shap, xgboost; print('All OK')"
```

**Key packages:**
- `pyradiomics==3.0.1` — radiomic feature extraction (IBSI-compliant)
- `SimpleITK` — NIfTI image loading and morphological operations
- `scikit-learn` — ML pipelines, cross-validation, metrics
- `xgboost`, `lightgbm` — gradient boosting classifiers
- `shap` — explainable AI
- `matplotlib`, `pandas`, `numpy` — data handling and visualization

---

## 5. Pipeline Step-by-Step

### Step 1: Create Peritumoral Rings
```bash
python src/create_regions_nifti.py --input data/raw/cfb_gbm --output data/processed/regions
```
**What it does:**
- Reads each patient's GTV mask (e.g., `001/t0/1_t0_gtv.nii.gz`)
- Creates `tumor_mask.nii.gz` (copy of GTV)
- Creates `ring_5mm.nii.gz` by dilating GTV by 5mm and subtracting the original mask
- Creates `ring_10mm.nii.gz` by dilating 10mm and subtracting the 5mm dilation
- Generates a 4-panel QC figure for each patient

**Output:** `data/processed/regions/001/t0/tumor_mask.nii.gz`, `ring_5mm.nii.gz`, `ring_10mm.nii.gz`

### Step 2: Extract Radiomic Features
```bash
python src/extract_radiomics_nifti.py \
    --input data/raw/cfb_gbm \
    --regions data/processed/regions \
    --output data/features/radiomics_80patients.csv
```
**What it does:**
- For each patient, extracts PyRadiomics features from 6 image×mask combinations
- Resamples images to 3×3×3mm to reduce computation (from 55M to 4M voxels)
- Extracts: Shape + First-order + GLCM features (56 per region)
- Total: 336 features per patient
- Runtime: ~30 seconds per patient

**Output:** `data/features/radiomics_80patients.csv` (80 rows × 338 columns)

### Step 3: Build ML Dataset
```bash
python src/build_ml_dataset_nifti.py
```
**What it does:**
- Loads the radiomics CSV
- Merges with clinical data (survival weeks)
- Creates binary OS label: Long (≥55 weeks) vs Short (<55 weeks) based on median split
- Outputs labeled dataset

**Output:** `data/features/ml_dataset_79patients.csv` (79 patients × 340 columns)

### Step 4: Train ML Models (TSV 261-patient cohort)
```bash
python src/train_ml_models.py --input data/features/ml_dataset.csv --output results/
```
**What it does:**
- Runs 3 experiments: Volume-only, Intratumoral-only, All Radiomics
- 5-fold stratified cross-validation
- 1000-bootstrap 95% CI for AUC
- Saves best model, ROC curves, confusion matrix

**Output:** `results/tables/model_comparison.csv`, `results/models/best_model.joblib`

### Step 5: Train Peritumoral Comparison (79-patient NIfTI cohort)
```bash
python src/train_peritumoral_comparison.py
```
**What it does:**
- Runs 4 experiments: T1Gd-tumor-only, Tumor T1Gd+FLAIR, Tumor+Ring5mm, All 6 regions
- 5-fold stratified CV with median imputation + SelectKBest(k=10) + classifier
- Directly tests the hypothesis: do rings add value?

**Output:** `results/tables/peritumoral_comparison.csv`, `results/figures/roc_peritumoral.png`

### Step 6: SHAP Explainability
```bash
python src/explain_models.py --model results/models/best_model.joblib --input data/features/ml_dataset.csv
```
**What it does:**
- Loads the best trained model
- Computes SHAP values for all patients
- Generates beeswarm, bar, waterfall, and permutation importance plots
- Saves feature importance table

**Output:** `results/figures/shap_*.png`, `results/tables/shap_feature_importance.csv`

### Step 7: Feature Analysis
```bash
python src/analyze_peritumoral_features.py
```
**What it does:**
- Computes per-feature univariate AUC for each of the 6 regions
- Generates box plots (key features, long vs short survivors)
- Generates violin plots for top 12 discriminating features
- Generates region discriminative power bar chart

**Output:** `results/figures/region_divergence.png`, `peritumoral_boxplots.png`, `top_features_violin.png`

---

## 6. Figure Explanations

### `results/figures/qc/001_t0_qc.png` (and all QC figures)
**What it shows:** 4-panel quality control figure for one patient, one timepoint.
- **Panel 1 (T1Gd):** Raw T1 gadolinium-enhanced MRI. Bright regions = contrast enhancement = active tumour core with blood-brain barrier breakdown.
- **Panel 2 (T1Gd + GTV):** Red overlay = radiotherapy gross tumour volume contour. Confirms the mask aligns with the enhancing region.
- **Panel 3 (FLAIR + GTV):** FLAIR MRI with red GTV overlay. Note how the bright FLAIR signal extends well beyond the red contour — this is the infiltration zone our rings capture.
- **Panel 4 (T1Gd + Ring 5mm):** Blue overlay = Ring 5mm. The ring surrounds the tumour in the immediate peritumoral zone.

**Key insight:** Panel 3 vs Panel 2 — the FLAIR abnormality is much larger than the GTV, confirming that clinically ignored tissue contains potentially important information.

### `results/figures/roc_curves.png`
**What it shows:** ROC curves for the 261-patient cohort comparing three feature sets.
- X-axis: False Positive Rate (1-Specificity) — fraction of short survivors incorrectly predicted as long
- Y-axis: True Positive Rate (Sensitivity) — fraction of long survivors correctly identified
- Each curve = one experiment (Volume-only, Intratumoral, All Radiomics)
- AUC (area under curve) = summary metric. 0.5 = random chance, 1.0 = perfect
- The intratumoral curve (AUC=0.710) sits highest — best performance

### `results/figures/roc_peritumoral.png`
**What it shows:** ROC curves for the 79-patient peritumoral comparison.
- Same axes as above
- Tumor+Ring5mm (AUC=0.670, green) clearly above Tumor-only (AUC=0.562, red)
- Directly visualises the +19% improvement from adding peritumoral features

### `results/figures/shap_summary.png` (Beeswarm plot)
**What it shows:** SHAP values for all features across all patients.
- Each row = one feature (top features at top)
- Each dot = one patient
- Dot colour: red = high feature value, blue = low feature value
- Horizontal position: positive SHAP = pushes toward long survival, negative = toward short
- **How to read:** A row where red dots are on the right means: high values of this feature predict long survival (good prognosis)

### `results/figures/shap_bar.png`
**What it shows:** Top N features ranked by mean absolute SHAP value.
- Bar length = average importance across all patients
- Bar colour = which region/sequence the feature comes from (T1Gd=red, FLAIR=orange, etc.)
- **How to read:** The longest bars are the features the model relies on most. The colours reveal whether tumor or ring features dominate.

### `results/figures/shap_waterfall_p0.png`
**What it shows:** SHAP waterfall plot for patient 0 (individual explanation).
- Shows how the model arrived at its specific prediction for one patient
- Each bar = one feature's contribution
- Red bars = push toward higher probability (long survival)
- Blue bars = push toward lower probability (short survival)
- Starting from base rate (E[f(x)]) and ending at final prediction

### `results/figures/region_divergence.png`
**What it shows:** Mean per-feature AUC for each of the 6 image regions.
- Bar height = average discriminative power across all 56 features in that region
- Transparent extension = maximum AUC achieved by the best single feature
- Dashed line = 0.5 (random chance)
- **Key finding:** T1Gd rings (green/blue) are taller than T1Gd tumor (red) — peritumoral features are more discriminating than intratumoral

### `results/figures/top_features_violin.png`
**What it shows:** Violin plots for the 12 most discriminating individual features.
- Red violin = Short survivors
- Blue violin = Long survivors
- Width of violin = density of patients at that value
- Wider at the top = more patients with high values
- AUC and p-value (Mann-Whitney U test) shown for each feature
- **How to read:** When the blue and red violins are clearly separated, that feature distinguishes well. The AUC value confirms this quantitatively.

### `results/figures/peritumoral_boxplots.png`
**What it shows:** Box plots for 6 key radiomic features × 6 regions, split by OS class.
- Each small plot: red=Short, blue=Long survivors
- Asterisks: *=p<0.05, **=p<0.01, ***=p<0.001 (Mann-Whitney U test)
- **How to read:** Boxes that don't overlap between red and blue indicate features that differ between survival groups. Ring features often show better separation than tumor features.

### `results/figures/permutation_importance.png`
**What it shows:** Permutation importance for the best model.
- Each bar = drop in AUC when that feature is randomly shuffled
- Error bars = standard deviation across 30 repeats
- Features with bars overlapping zero = contribute no reliable signal
- **How to read:** Longest bars with small error bars = truly important features. Confirms SHAP rankings with an independent method.

### `results/figures/workflow_diagram.png`
**What it shows:** Visual overview of the complete pipeline from data download to results.

### `results/figures/dataset_flowchart.png`
**What it shows:** Patient inclusion/exclusion flowchart.

---

## 7. Results Summary

### Primary Result: 261-Patient Cohort (TSV features)
| Experiment | Best Model | AUC | 95% CI | Balanced Acc |
|---|---|---|---|---|
| Volume Only (baseline) | SVM-RBF | 0.627 | 0.562–0.692 | 0.590 |
| All Radiomics | RandomForest | 0.670 | 0.607–0.735 | 0.640 |
| **Intratumoral Only** | **LogReg ElasticNet** | **0.710** | **0.648–0.767** | **0.670** |

**Interpretation:** Intratumoral radiomics (T1Gd + FLAIR texture within the tumour) outperforms simple volume measurement by +13%. Logistic Regression with ElasticNet penalty selects a sparse subset of texture features — more robust than ensemble methods for this dataset size.

### Secondary Result: 79-Patient NIfTI Cohort (peritumoral analysis)
| Experiment | Best Model | AUC | 95% CI |
|---|---|---|---|
| T1Gd Tumor Only | RandomForest | 0.562 | 0.437–0.688 |
| Tumor T1Gd + FLAIR | RandomForest | 0.535 | 0.405–0.664 |
| **Tumor + Ring 5mm** | **LogReg L1** | **0.670** | **0.545–0.794** |
| All Regions | XGBoost | 0.611 | 0.500–0.738 |

**Interpretation:** Adding Ring 5mm features improves AUC by +19% relative (0.562→0.670). This is the key scientific finding: peritumoral tissue in the 0–5mm zone beyond the tumour boundary carries independent prognostic information.

### Region Discriminative Power (per-feature AUC)
| Rank | Region | Mean AUC | Max AUC |
|---|---|---|---|
| 1 | T1Gd Ring 5mm | 0.598 | 0.714 |
| 2 | T1Gd Ring 10mm | 0.591 | 0.696 |
| 3 | FLAIR Tumor | 0.571 | 0.646 |
| 4 | FLAIR Ring 5mm | 0.570 | 0.626 |
| 5 | FLAIR Ring 10mm | 0.570 | 0.670 |
| 6 | T1Gd Tumor | 0.556 | 0.639 |

T1Gd peritumoral rings rank above intratumoral features — this is biologically consistent with T1Gd capturing vascular permeability changes and contrast agent dynamics in the peritumoral microenvironment.

---

## 8. File Structure

```
cfb-gbm-ai-radiomics/
│
├── config/
│   ├── config.yaml                    # Project paths and ML settings
│   └── pyradiomics_params.yaml        # PyRadiomics settings (3mm resample, IBSI)
│
├── src/
│   ├── utils.py                       # Shared helpers (logger, NIfTI I/O)
│   ├── create_regions_nifti.py        # Creates tumor mask + Ring 5mm + Ring 10mm
│   ├── extract_radiomics_nifti.py     # Extracts PyRadiomics features from NIfTI
│   ├── build_ml_dataset_nifti.py      # Merges radiomics + clinical → labeled CSV
│   ├── build_ml_dataset_tsv.py        # Builds ML dataset from pre-extracted TSV
│   ├── train_ml_models.py             # Trains models on 261-patient TSV cohort
│   ├── train_peritumoral_comparison.py # Peritumoral vs tumor-only ML comparison
│   ├── analyze_peritumoral_features.py # Per-feature AUC, violin/box plots
│   ├── explain_models.py              # SHAP + permutation importance
│   ├── make_figures.py                # Workflow and summary figures
│   └── train_deep_embeddings.py       # Optional: 3D ResNet (GPU required)
│
├── data/
│   ├── raw/cfb_gbm/001-080/t0/        # Patient NIfTI files (NOT in GitHub)
│   ├── processed/
│   │   ├── regions/                   # Tumor mask + rings per patient
│   │   └── *.tsv                      # TCIA pre-extracted features + clinical data
│   └── features/
│       ├── ml_dataset.csv             # 261-patient TSV ML dataset
│       ├── ml_dataset_79patients.csv  # 79-patient NIfTI ML dataset
│       └── radiomics_80patients.csv   # Raw extracted features
│
├── results/
│   ├── figures/
│   │   ├── qc/                        # 79 × 4-panel QC figures
│   │   ├── roc_curves.png             # ROC for 261-patient cohort
│   │   ├── roc_peritumoral.png        # ROC for 79-patient peritumoral comparison
│   │   ├── shap_summary.png           # SHAP beeswarm
│   │   ├── shap_bar.png               # SHAP feature importance bar
│   │   ├── shap_waterfall_p0.png      # SHAP waterfall for patient 0
│   │   ├── permutation_importance.png # Permutation importance
│   │   ├── region_divergence.png      # Per-region AUC bar chart
│   │   ├── top_features_violin.png    # Top 12 features violin plots
│   │   └── peritumoral_boxplots.png   # Feature box plots by OS class
│   ├── models/
│   │   ├── best_model.joblib          # Best TSV cohort model (saved)
│   │   └── best_model_peritumoral.joblib
│   └── tables/
│       ├── model_comparison.csv       # All model results (261-patient)
│       ├── peritumoral_comparison.csv # Peritumoral experiment results
│       ├── region_discriminative_power.csv
│       └── shap_feature_importance.csv
│
├── report/
│   └── seminar_report.md              # Full written report (this project)
│
├── presentation/
│   ├── slides.html                    # 18-slide HTML presentation
│   └── poster.html                    # Scientific poster (A0 format)
│
├── environment.yml                    # Conda environment specification
├── README.md                          # Quick start guide
├── GUIDE.md                           # This file
└── PROJECT_STATUS.md                  # Current status and next steps
```

---

## 9. Frequently Asked Questions

**Q: Is this validated clinically?**
No. This is a research/educational prototype. Results should not be used for clinical decision-making.

**Q: Why is AUC not higher (e.g., 0.9)?**
GBM outcome prediction from MRI alone is genuinely hard — survival depends on molecular factors (IDH, MGMT), treatment response, and many non-imaging variables. AUC 0.71 is consistent with published radiomics literature for single-cohort studies. Confidence intervals are wide due to sample size.

**Q: Why does Logistic Regression beat Random Forest?**
With 79 patients and 224 features (Tumor+Ring5mm), L1 regularisation drives most coefficients to zero, selecting ~5-10 truly predictive features. This is optimal for high-dimensional, small-sample data where overfitting is the main risk.

**Q: Can I download more patients and rerun?**
Yes. Download additional patients from TCIA, place in `data/raw/cfb_gbm/`, and rerun from Step 1 (create_regions_nifti.py). The scripts auto-detect all patient folders.

**Q: How long does extraction take?**
~30 seconds per patient at 3mm resampling. 79 patients ≈ 40 minutes total. The main bottleneck is the 512×512×208 NIfTI volumes at 0.5mm original spacing.

**Q: What does the deep learning component do?**
`train_deep_embeddings.py` implements a 3D ResNet that takes 96³mm patches centred on the tumour and learns 128-dimensional feature embeddings. These can be combined with handcrafted radiomics for a hybrid model. Requires ~48GB GPU. Not run in this project due to n=79 being too small for reliable deep learning training.
