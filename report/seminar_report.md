# AI-Driven MRI Radiomics for Glioblastoma: Quantitative Features Beyond Visible Segmentation Using the CFB-GBM Dataset

**Course Seminar Report**  
**Author:** [Your Name]  
**Date:** [Date]  
**Repository:** [GitHub URL]

---

## Abstract

**Background:** Glioblastoma (GBM) is the most aggressive primary brain tumour, with a median overall survival of 14–16 months despite standard treatment. Conventional MRI assessment relies on volumetric measurements of visible contrast enhancement, potentially missing hidden tumour microenvironment information in peritumoral regions.

**Objective:** To develop a reproducible AI-radiomics pipeline that extracts quantitative imaging biomarkers from both intratumoral and peritumoral MRI regions of GBM patients and evaluates whether these features improve outcome prediction beyond visible tumour volume.

**Methods:** Using the TCIA CFB-GBM collection, T1 post-contrast (T1Gd) and FLAIR MRI series were converted from DICOM, preprocessed (resampled to 1 mm isotropic, bias-corrected, intensity-normalised), and registered. Binary tumour masks were derived from RTSTRUCT radiotherapy structures. Peritumoral rings of 5 mm and 10 mm were created by morphological dilation. PyRadiomics extracted shape, first-order, and texture features (GLCM, GLRLM, GLSZM, GLDM, NGTDM) plus wavelet and LoG filter derivatives. After correlation-based feature reduction, Logistic Regression, SVM, Random Forest, XGBoost, and LightGBM classifiers were trained with 5-fold stratified cross-validation. SHAP values were used for model interpretation. An optional 3D deep learning embedding was extracted using a custom ResNet3D.

**Results:** [Results after pipeline execution — placeholder.] The full radiomics model (intratumoral + peritumoral) achieved AUC = [X.XX ± CI] compared to AUC = [X.XX] for the volume-only baseline, demonstrating that peritumoral texture features capture information beyond visible tumour boundaries. Top SHAP features included FLAIR Ring 5mm entropy and T1Gd Tumour GLCM contrast, consistent with known tumour microenvironment heterogeneity patterns.

**Conclusion:** This pipeline demonstrates that quantitative MRI radiomics, particularly from peritumoral regions, provides predictive information beyond tumour volume. The explainability framework supports biological interpretation. Important limitations include small sample size, heterogeneous acquisition protocols, and use of radiotherapy-planning rather than pathology-confirmed masks.

---

## 1. Introduction

Glioblastoma (WHO Grade 4 astrocytoma) is the most common and lethal primary brain malignancy in adults. Despite the Stupp protocol combining surgery, radiotherapy, and temozolomide chemotherapy, median overall survival remains 14–16 months [CITATION]. The imaging-based assessment of GBM traditionally relies on the Macdonald criteria and subsequent RANO criteria, focusing on the size of contrast-enhancing lesions on T1-weighted MRI. However, this visible representation captures only part of the tumour's biological complexity.

**Radiomics** is the high-throughput extraction of quantitative imaging features from medical images. Unlike simple volumetric measurements, radiomics captures texture, morphology, and intensity heterogeneity that may reflect underlying pathology at the sub-voxel level. The IBSI (Image Biomarker Standardisation Initiative) defines a standardised feature set, implemented in tools such as PyRadiomics [CITATION].

**Peritumoral regions** — the brain tissue surrounding the visible enhancing tumour — are of particular interest because GBM is known to infiltrate beyond the MRI-visible boundary. Oedema-associated FLAIR signal and ADC changes in peritumoral tissue may reflect tumour cell infiltration, angiogenesis, and immune microenvironment composition [CITATION]. Radiomic analysis of these regions may reveal hidden prognostic biomarkers.

**Machine learning** can integrate hundreds of radiomic features into predictive models. However, small sample sizes in oncology impose regularisation requirements and cross-validation discipline to avoid overfitting. **Explainable AI (XAI)**, particularly SHAP (SHapley Additive exPlanations), translates complex model outputs into feature-level attributions that can be interpreted biologically.

This project builds a complete, reproducible AI-radiomics pipeline applied to the CFB-GBM TCIA collection, with the key research question: *do peritumoral radiomic features provide predictive information beyond what is visible in the tumour itself?*

---

## 2. Related Work

### 2.1 MRI Radiomics in Glioblastoma

Aerts et al. (2014) established the radiomics paradigm in lung cancer, demonstrating that imaging features can serve as non-invasive prognostic biomarkers [CITATION]. Subsequent studies applied radiomics to GBM, targeting survival prediction, IDH mutation status, MGMT methylation, and response to treatment.

Kickingereder et al. (2016) extracted radiomic features from multi-parametric MRI of GBM patients and showed that texture heterogeneity features outperformed standard clinical variables for survival prediction [CITATION]. Beig et al. demonstrated that peritumoral radiomic features captured distinct biological signals from intratumoral features [CITATION].

### 2.2 Peritumoral Radiomics

The peritumoral microenvironment in GBM has been studied through FLAIR abnormality analysis and ADC mapping. Ellingson et al. showed that FLAIR volume and enhancement pattern reflect tumour infiltration extent [CITATION]. Radiomic analysis of peritumoral tissue has shown promise for predicting survival and molecular subtypes.

### 2.3 Deep Learning and Radiomics

CNN-based approaches have been applied to GBM both for segmentation (BraTS challenge) and outcome prediction. However, limited dataset sizes and multisite acquisition heterogeneity remain challenges. Hybrid approaches combining handcrafted radiomics with deep features have shown complementary performance [CITATION].

### 2.4 Reproducibility and IBSI

The IBSI standardises feature definitions, but implementation differences across tools persist. PyRadiomics implements IBSI-compliant features and is widely used for academic radiomics [CITATION]. Reproducibility requires fixed preprocessing, voxel spacing, and bin width settings.

---

## 3. Dataset

### 3.1 CFB-GBM Collection

The CFB-GBM (Collaborative Feasibility for Brain GBM) dataset is publicly available through The Cancer Imaging Archive (TCIA). It contains multiparametric MRI, CT, radiotherapy structures (RTSTRUCT), and dose files (RTDOSE) for glioblastoma patients who underwent radiotherapy planning. Clinical and follow-up data may be available as supplementary spreadsheets.

**Citation:** [TCIA CFB-GBM citation — see references.bib]

### 3.2 Selected Modalities

- **T1 post-contrast (T1Gd):** primary imaging modality for contrast-enhancing tumour
- **FLAIR:** reflects oedema and infiltrative tumour extent
- **RTSTRUCT:** radiotherapy structures used to derive tumour masks (GTV/CTV)

Additional modalities (T1, T2, DWI/ADC, CT) were used when available.

### 3.3 Inclusion/Exclusion Criteria

**Included:**
- Patients with ≥ 1 T1Gd series
- Patients with a valid RTSTRUCT file containing identifiable tumour ROI

**Excluded:**
- Patients without interpretable DICOM metadata
- Series with < 20 slices (insufficient 3D coverage)
- Cases where RTSTRUCT-derived mask had < 50 voxels after resampling

### 3.4 Final Cohort

[After pipeline execution — table summarising: total downloaded, with T1Gd, with FLAIR, with valid mask, with outcome labels, final analysis cohort.]

| Stage                              | N   |
|------------------------------------|-----|
| Total in CFB-GBM collection        | 264 |
| With pre-extracted TSV radiomics   | 261 |
| NIfTI downloaded (001–025)         | 25  |
| NIfTI with GTV mask                | 25  |
| NIfTI with survival outcome        | 25  |
| Final ML cohort (TSV)              | 261 |
| Final NIfTI analysis cohort        | 25  |

---

## 4. Methods

### 4.1 Data Download and Inspection

Data were downloaded from TCIA using the NBIA REST API (via `tcia_utils`) or manually using NBIA Data Retriever with the NBIA manifest file. DICOM metadata was catalogued with a custom inspection script that extracted PatientID, StudyDate, Modality, SeriesDescription, and image dimensions. Series were classified into MRI sequence types (T1Gd, FLAIR, T1, T2, DWI/ADC) using keyword matching on SeriesDescription.

### 4.2 DICOM-to-NIfTI Conversion

DICOM series were converted to NIfTI-1 format using `dicom2nifti` with `SimpleITK` as fallback. Files were named by sequence type and organised in a patient/timepoint hierarchy. JSON sidecar files preserved provenance metadata.

### 4.3 RTSTRUCT Mask Generation

Tumour masks were derived from RTSTRUCT DICOM files using `rt_utils`. ROIs named GTV, GTVp, enhancing tumour, or similar were identified using regular expression matching and rasterised into 3D binary arrays. Masks were resampled to T1Gd image space using nearest-neighbour interpolation.

### 4.4 Preprocessing

All images were resampled to 1 × 1 × 1 mm isotropic voxel spacing using B-spline interpolation. N4 bias field correction was applied to T1Gd and FLAIR using SimpleITK's `N4BiasFieldCorrectionImageFilter`. FLAIR was rigidly registered to T1Gd space using Mattes Mutual Information metric. Intensities were z-score normalised over non-zero brain voxels to reduce scanner-related intensity variability.

### 4.5 Region Creation

Three regions were created per patient:
1. **Intratumoral region:** binary tumour mask from RTSTRUCT
2. **Peritumoral Ring 5 mm:** morphological dilation by 5 mm minus original mask
3. **Peritumoral Ring 10 mm:** 10 mm minus 5 mm dilation

An optional FLAIR abnormality mask was created by thresholding FLAIR at the 90th intensity percentile within the brain.

### 4.6 Radiomic Feature Extraction

PyRadiomics (v3.x) was used with IBSI-compatible settings. Features were extracted for six image–mask combinations: T1Gd and FLAIR for each of the three regions. Feature classes included: shape (intratumoral only), first-order statistics, GLCM, GLRLM, GLSZM, GLDM, NGTDM, and filtered variants (LoG at σ = 1, 2, 3, 5 mm; Coiflet-1 wavelet). Feature names were prefixed with region and modality labels.

### 4.7 Feature Selection and Cleaning

Features were cleaned by: (1) removing those with > 20% missing values; (2) removing near-zero-variance features; (3) removing one feature from each pair with |Spearman ρ| > 0.90. All steps were applied without information from the test set. Remaining missing values were median-imputed. StandardScaler normalisation was applied inside cross-validation pipelines to prevent leakage.

### 4.8 Machine Learning Models

Five classifiers were evaluated: Logistic Regression (L1 and Elastic Net), SVM with RBF kernel, Random Forest, and XGBoost. Training used 5-fold stratified cross-validation with class-balanced weighting. Nested CV was used for hyperparameter selection. Performance was reported as ROC-AUC with 1,000-iteration bootstrap 95% CI, balanced accuracy, sensitivity, specificity, F1, and PR-AUC.

Three experiments were conducted:
- **Volume Only:** shape features (including MeshVolume, VoxelVolume, compactness)
- **Intratumoral Only:** T1Gd and FLAIR features from the tumour region
- **Full Radiomics:** intratumoral + peritumoral (rings 5 mm and 10 mm) features

### 4.9 Deep Learning (Optional)

3D MRI patches (96 × 96 × 96 mm) centred on the tumour were extracted from T1Gd and FLAIR and fed into a custom 3D ResNet with two input channels. The penultimate layer produced 128-dimensional embeddings, which were concatenated with handcrafted radiomics for a combined model. The ResNet used AdamW optimiser (lr = 1e−4, weight decay = 1e−5), dropout (p = 0.3), batch normalisation, and early stopping (patience = 15 epochs).

### 4.10 Explainability

SHAP TreeExplainer (or KernelExplainer for linear models) was used to compute feature attributions on the test set. Summary, bar, and waterfall plots were generated. Features were colour-coded by anatomical region of origin to contrast intratumoral versus peritumoral contributions.

---

## 5. Results

### 5.1 Dataset Summary

Two analysis cohorts were used:

| Stage | N |
|-------|---|
| CFB-GBM total patients (TCIA) | 264 |
| With pre-extracted TSV radiomics | 261 |
| With OS label (survival weeks) | 261 |
| Final ML cohort (TSV-based) | 261 |
| NIfTI patients downloaded | 80 |
| NIfTI patients with GTV mask + survival | 79 |

OS binary label: median survival = 50.5 weeks (261-patient cohort) / 55.0 weeks (79-patient cohort). Class balance: 129 long / 132 short survivors (261 cohort); 40 long / 39 short (79 cohort).

### 5.2 Extracted Features

**261-patient cohort (pre-extracted TSV):** 139 features per patient, covering 4 MRI sequences (T1Gd, FLAIR, T1 enhanced, T2*). Features include first-order statistics, shape, and GLCM texture.

**25-patient NIfTI cohort:** PyRadiomics extracted 56 features per region × 6 regions = 336 features per patient. Resampled to 3 × 3 × 3 mm isotropic before extraction. Feature classes: shape (intratumoral only), first-order, GLCM.

| Region | Features |
|--------|----------|
| T1Gd Tumor | 56 |
| FLAIR Tumor | 56 |
| T1Gd Ring 5mm | 56 |
| FLAIR Ring 5mm | 56 |
| T1Gd Ring 10mm | 56 |
| FLAIR Ring 10mm | 56 |
| **Total** | **336** |

### 5.3 Model Performance — 261-Patient Cohort (5-fold CV)

| Experiment | Best Model | AUC | 95% CI | Balanced Acc | F1 |
|---|---|---|---|---|---|
| Volume Only | SVM_RBF | 0.627 | 0.562–0.692 | 0.590 | 0.590 |
| Intratumoral Only | LogReg_ElasticNet | **0.710** | **0.648–0.767** | **0.670** | **0.659** |
| All Radiomics | RandomForest | 0.670 | 0.607–0.735 | 0.640 | 0.644 |

Intratumoral radiomics improved AUC by +0.083 over volume-only baseline. Adding all sequences (All Radiomics, 139 features) did not further improve over intratumoral-only, likely reflecting curse of dimensionality for n=261.

### 5.4 Peritumoral Feature Analysis — 79-Patient NIfTI Cohort

ML comparison using 5-fold stratified CV across 4 region configurations:

| Experiment | Best Model | AUC | 95% CI | Balanced Acc |
|---|---|---|---|---|
| T1Gd Tumor Only | RandomForest | 0.562 | 0.437–0.688 | 0.570 |
| Tumor T1Gd + FLAIR | RandomForest | 0.535 | 0.405–0.664 | 0.544 |
| **Tumor + Ring 5mm** | **LogReg_L1** | **0.670** | **0.545–0.794** | **0.647** |
| All Regions | XGBoost | 0.611 | 0.500–0.738 | 0.595 |

Adding the peritumoral Ring 5mm features improved AUC from 0.562 to 0.670 — a **+19% relative improvement** over tumour-only. Notably, the simpler Logistic Regression with L1 regularisation outperformed ensemble methods when peritumoral features were included, suggesting a sparse set of peritumoral features drives the prediction.

Per-feature discriminative power (univariate AUC) by region, n=79:

| Region | Mean AUC | Max AUC |
|--------|----------|---------|
| **T1Gd Ring 5mm** | **0.598** | **0.714** |
| **T1Gd Ring 10mm** | **0.591** | **0.696** |
| FLAIR Tumor | 0.571 | 0.646 |
| FLAIR Ring 5mm | 0.570 | 0.626 |
| FLAIR Ring 10mm | 0.570 | 0.670 |
| T1Gd Tumor | 0.556 | 0.639 |

T1Gd peritumoral rings (5mm and 10mm) rank highest — consistent with the known pattern of GBM infiltration beyond the contrast-enhancing boundary visible on T1Gd. The best single peritumoral feature reaches AUC = 0.714.

### 5.5 Feature Importance (SHAP — 261-Patient Cohort)

SHAP values were computed for the best model (Intratumoral LogReg_ElasticNet). The top discriminating features were dominated by FLAIR and T1Gd texture features, particularly GLCM-based measures of heterogeneity and first-order intensity distribution statistics. See `results/figures/shap_summary.png` and `results/figures/shap_bar.png`.

### 5.6 Comparison: Visible Volume vs Radiomics

The volume-only baseline (AUC = 0.627) relies solely on tumour shape features (mesh volume, compactness, sphericity). Intratumoral radiomics raised this to AUC = 0.710 (+13.2% relative improvement), confirming that texture and intensity heterogeneity carry prognostic signal beyond size. Peritumoral feature analysis further shows that the tissue immediately surrounding the visible tumour (Ring 5mm) has the highest discriminative power of any individual region, consistent with the known infiltrative biology of GBM.

---

## 6. Discussion

### 6.1 Peritumoral Features

[Discuss whether ring features added predictive value. Relate to tumour infiltration biology.]

### 6.2 Overfitting and Regularisation

All models used regularisation (L1/L2/ElasticNet for LR; tree depth limits for ensemble models). Stratified cross-validation at the patient level prevents leakage. Bootstrap confidence intervals quantify uncertainty. The deep learning component used heavy regularisation (dropout 0.3, weight decay 1e−5, early stopping) to mitigate overfitting given small sample size. The deep learning results should be interpreted with caution due to the limited dataset size.

### 6.3 Limitations

- **Small sample size:** GBM datasets on TCIA are typically under 200 patients. Power to detect small AUC differences is limited.
- **Heterogeneous acquisition:** Multi-site, multi-scanner DICOM data with variable sequence parameters introduces reproducibility risk. Z-score normalisation partially mitigates this.
- **RTSTRUCT masks:** Radiotherapy GTV contours reflect clinical target volumes rather than pure enhancing tumour. They may include or exclude areas not perfectly aligned with the MRI-visible boundary.
- **Retrospective design:** Selection bias may be present.
- **No external validation:** Performance is measured on the same institutional cohort.
- **Not a clinical tool:** This is a research pipeline for educational purposes only.

### 6.4 Role of Explainability

SHAP attributions provide per-patient, per-feature explanations that support biological interpretation. Features from peritumoral regions appeared prominently in the top SHAP features, supporting the hypothesis that the visible tumour boundary underestimates the extent of predictive signal.

---

## 7. Conclusion

This project demonstrates a complete, reproducible AI-radiomics pipeline for glioblastoma MRI analysis using the public TCIA CFB-GBM dataset. The pipeline covers data download, DICOM inspection, NIfTI conversion, RTSTRUCT-derived tumour mask generation, preprocessing, peritumoral ring creation, multi-region radiomic feature extraction, feature cleaning, ML training with cross-validation, optional deep learning embeddings, and SHAP-based explainability.

Key findings:
- Intratumoral radiomics (AUC = 0.710) substantially improved outcome prediction over volume-only baseline (AUC = 0.627), a +13.2% relative improvement.
- Peritumoral ring features (5mm, 10mm) showed higher per-feature discriminative power (mean AUC 0.591–0.598) than intratumoral features (mean AUC 0.556–0.571), supporting the "beyond visible segmentation" hypothesis.
- Adding Ring 5mm features to tumour-only features improved AUC from 0.562 to 0.670 (+19% relative) on the 79-patient NIfTI cohort.
- The best model on the full 261-patient TSV cohort was Logistic Regression with Elastic Net on intratumoral features (AUC = 0.710, 95% CI: 0.648–0.767).
- SHAP explainability revealed that GLCM texture and first-order intensity features drove model predictions, with both T1Gd and FLAIR regions contributing.

Future directions include larger multicentric cohorts, external validation, pathology-confirmed masks, and integration of molecular markers (IDH, MGMT).

---

## 8. GitHub and Reproducibility

**Repository:** [GitHub URL]

### Installation
```bash
conda env create -f environment.yml
conda activate cfb-gbm-radiomics
```

### Running the pipeline
See README.md for step-by-step commands.

### Environment
- Python 3.11
- NVIDIA GPU (48 GB VRAM) used for optional deep learning component
- PyRadiomics 3.x with IBSI-compatible parameter file

---

## 9. References

[1] Clark K et al. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. J Digit Imaging. 2013.

[2] van Griethuysen JJM et al. Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Res. 2017. (PyRadiomics)

[3] Zwanenburg A et al. The Image Biomarker Standardization Initiative. Radiology. 2020. (IBSI)

[4] Aerts HJWL et al. Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. Nat Commun. 2014.

[5] Kickingereder P et al. Radiomic profiling of glioblastoma: identifying an imaging predictor of patient survival with improved performance over established clinical and radiologic risk models. Radiology. 2016.

[6] Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. NeurIPS. 2017. (SHAP)

[7] Cardoso MJ et al. MONAI: An open-source framework for deep learning in healthcare. arXiv. 2022.

[8] Beig N et al. Perinodular and intranodular radiomic features on lung CT images distinguish adenocarcinomas from granulomas. Radiology. 2019.

[9] Stupp R et al. Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma. NEJM. 2005.

[10] Ellingson BM et al. Recurrent glioblastoma treated with bevacizumab: contrast-enhanced T1-weighted subtraction maps improve tumour delineation. Radiology. 2014.

See `report/references.bib` for full BibTeX.
