# Slide Outline: AI-Driven MRI Radiomics for Glioblastoma

**Presentation target:** 15–20 minutes + 5 minutes Q&A  
**Total slides:** ~18

---

## Slide 1 — Title
**AI-Driven MRI Radiomics for Glioblastoma**  
*Quantitative Features Beyond Visible Segmentation*

- Course / institution
- Your name, date
- Background image: T1Gd MRI with tumour mask overlay

---

## Slide 2 — Motivation
**Why is GBM Hard to Evaluate Visually?**

- GBM is aggressive: median OS 14–16 months (Stupp 2005)
- Current standard: RANO criteria — measure contrast-enhancing volume
- Problem: tumour infiltrates beyond visible boundary
- Visual assessment misses intratumour heterogeneity and peritumoral microenvironment
- Figure: T1Gd image with visible enhancement vs FLAIR showing broader oedema

---

## Slide 3 — What is Radiomics?
**From Pixels to Quantitative Biomarkers**

- Radiomics = high-throughput extraction of quantitative features from medical images
- Features capture: shape, intensity distribution, texture, spatial patterns
- IBSI standardises definitions across tools
- PyRadiomics: open-source, reproducible, IBSI-compliant
- Figure: radiomics feature class diagram (from make_figures.py)

---

## Slide 4 — Project Goal and Dataset
**Research Question**

> Do peritumoral radiomic features capture predictive information  
> that is not visible in the tumour itself?

- Dataset: TCIA CFB-GBM collection
- Modalities: T1Gd + FLAIR + RTSTRUCT masks
- Public dataset — citation required, raw data not included in repo
- Cohort: N patients (fill in after pipeline run)

---

## Slide 5 — Pipeline Overview
**Complete Reproducible Pipeline**

- Figure: workflow_diagram.png
- Narrate each stage: download → inspect → convert → mask → preprocess → regions → radiomics → ML → XAI → report

---

## Slide 6 — Data Download and Inspection
**TCIA CFB-GBM Dataset**

- Option A: NBIA Data Retriever + manifest file
- Option B: Programmatic via NBIA REST API + `tcia_utils`
- Metadata CSV: patient, study, series, modality, sequence type
- Summary table: N patients, N with T1Gd, FLAIR, RTSTRUCT
- Figure: completeness matrix heatmap

---

## Slide 7 — Preprocessing Pipeline
**Image Preparation Steps**

1. DICOM → NIfTI (dicom2nifti / SimpleITK)
2. RTSTRUCT → Binary mask (rt_utils)
3. Resample to 1 mm isotropic
4. N4 bias field correction (SimpleITK)
5. FLAIR → T1Gd registration (Mattes MI)
6. Z-score normalisation (non-zero voxels)

- Figure: before/after N4 correction example (or single-patient montage)

---

## Slide 8 — Beyond Visible Segmentation: Peritumoral Rings
**The Key Innovation**

- Intratumoral region: GTV from RTSTRUCT
- Peritumoral Ring 5 mm: morphological dilation − tumour mask
- Peritumoral Ring 10 mm: outer shell
- Figure: T1Gd + tumour mask, FLAIR + ring 5mm (from QC figures)

**Why peritumoral matters:**
- GBM infiltrates beyond enhancement
- Oedema and vasogenic changes in FLAIR reflect microenvironment
- Cell density, angiogenesis, hypoxia signals in peritumoral tissue

---

## Slide 9 — Radiomic Feature Extraction
**PyRadiomics: Feature Classes**

- 6 image–mask combinations extracted
- Feature classes: shape, first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM
- Filtered images: LoG (4 σ), wavelet (8 bands)
- Total raw features: ~N (fill in)
- Feature naming: `T1Gd_Tumor_original_glcm_Contrast`

---

## Slide 10 — Feature Selection
**Avoiding Overfitting through Dimensionality Reduction**

- Remove features with > 20% missing
- Remove near-zero variance features
- Remove correlated features |Spearman ρ| > 0.90
- All steps inside cross-validation (no leakage)
- Figure: correlation heatmap before/after reduction

---

## Slide 11 — Machine Learning Models
**Comparing 5 Classifiers × 3 Feature Sets**

Experiments:
1. Volume Only (shape features — baseline)
2. Intratumoral radiomics (T1Gd + FLAIR tumour)
3. Full radiomics (intratumoral + peritumoral)

Models: Logistic Regression (L1/ElasticNet), SVM-RBF, Random Forest, XGBoost, LightGBM

Evaluation: 5-fold stratified CV, AUC + 1000-iteration bootstrap CI

---

## Slide 12 — Results: Model Comparison
**The Key Finding**

- Figure: model_comparison_bar.png (AUC ± CI for each experiment)
- Table: best model per experiment
- Key message: did peritumoral features improve over volume-only?
- (Fill in after pipeline run)

---

## Slide 13 — ROC Curves
**Performance Visualisation**

- Figure: roc_curves.png
- Discuss AUC, sensitivity/specificity trade-off
- Note class imbalance handling (balanced weighting)
- Confidence intervals reflect uncertainty at small sample sizes

---

## Slide 14 — Optional: Deep Learning Component
**3D ResNet Embeddings (GPU-accelerated)**

- 3D patches (96³ mm) centred on tumour mask centroid
- 2-channel input: T1Gd + FLAIR
- ResNet3D: Conv stem → 3 residual blocks → AdaptiveAvgPool → FC(128)
- Dropout 0.3, BatchNorm, AdamW, early stopping
- Embeddings concatenated with handcrafted radiomics for combined model
- Figure: training curves (loss / AUC vs epoch)
- Discuss overfitting risk with small N

---

## Slide 15 — Explainable AI: SHAP
**Understanding Model Predictions**

- Figure: shap_summary.png (beeswarm)
- Figure: shap_bar.png (top features coloured by region)
- Key insight: which regions contribute most?
- Biological interpretation (cautious):
  - FLAIR Ring entropy → texture disorder in peritumoral microenvironment
  - T1Gd Tumour GLCM contrast → intratumoral vascular heterogeneity

---

## Slide 16 — Feature Interpretation
**What Do These Features Mean?**

| Feature Type | Region | Possible Biology |
|---|---|---|
| FLAIR Ring 5mm Entropy | Peritumoral | Oedema heterogeneity, infiltration |
| T1Gd Tumour GLCM Contrast | Intratumoral | Vascular permeability pattern |
| T1Gd Ring GLRLM RunNonUnif | Peritumoral | Texture anisotropy |
| Shape: Compactness | Intratumoral | Tumour margin irregularity |

**Important caveat:** correlations, not causal mechanisms. Interpretation requires domain expertise and validation.

---

## Slide 17 — Limitations and Future Work
**Honest Assessment**

Limitations:
- Small sample size → high uncertainty, wide CIs
- Single cohort, no external validation
- RTSTRUCT masks are radiotherapy targets, not pure pathology
- Heterogeneous MRI acquisition across patients
- Not a clinical tool — educational/research prototype

Future work:
- Larger datasets (BraTS, TCIA TCGA-GBM)
- Prospective collection with standardised protocols
- Pathology-confirmed masks
- Integration of molecular markers (IDH, MGMT)
- External validation

---

## Slide 18 — Conclusion and GitHub
**Summary**

- Built a reproducible AI-radiomics pipeline for GBM MRI
- Peritumoral rings capture information beyond visible tumour boundary
- Explainable ML (SHAP) interprets feature contributions
- [Key AUC result — fill in]
- All code available on GitHub

**Repository:** [URL]

```
conda env create -f environment.yml
conda activate cfb-gbm-radiomics
python src/inspect_dicom.py ...
```

**Thank you for your attention.**

---

## Appendix Slides (backup)

### A — DICOM Tags Used for Classification
### B — PyRadiomics Parameter File (key settings)
### C — Cross-Validation Strategy Diagram
### D — Patient-Level Split (preventing leakage)
### E — Full Model Comparison Table
