# Quarantine Area: BD-KDD Chronic Kidney Disease (CKD) Dataset

## 1. Audit Overview
This directory preserves all raw inputs, processed matrices, configurations, and experimental outputs for the **BD-KDD Chronic Kidney Disease (CKD)** dataset, which has been removed from the active DODA primary benchmark.

## 2. Dataset Lineage & Citation
- **Originating Publication**: *BD-KDD: A real-world clinical dataset for kidney disease diagnosis and healthy classification*, Data in Brief (Elsevier), Volume 66, June 2026, Article 112738.
- **DOI**: `10.1016/j.dib.2026.112738`
- **Institution**: Popular Diagnostic Center (Savar Branch, Dhaka, Bangladesh).
- **Public Archive**: Zenodo (*BD-KDD: A Clinical Dataset of Kidney Disease Diagnosis*).
- **Misattribution Origin**: Initial setup labeled this file as the Apollo Hospitals / UCI CKD benchmark (Soundarapandian et al., 2015). The canonical UCI CKD dataset has $N=400$ instances with missing values; this file has $N=988$ complete records.

## 3. Preserved Raw Data & Verification
- **Raw File**: `data/quarantine/ckd_raw.csv`
- **SHA-256 Checksum**: `cb9d92f77f20859edaf73ba9236d8fed814e854016ab2eaf83e13ddba45bfaeb`
- **Row Count**: 988 records
- **Class Balance**: 481 control (0), 507 kidney disease (1)

## 4. Reason for Quarantine (Forensic Audit Findings)
1. **Zero Ground-Truth Biomarker Separation**:
   - Under KDIGO guidelines, Serum Creatinine (`sc`) and Urine Albumin (`al`) are definitive diagnostic criteria. In this dataset:
     - `sc`: Class 0 mean = $7.41$ mg/dL vs Class 1 mean = $7.65$ mg/dL ($p = 0.359$, univariate ROC-AUC = $0.518$).
     - `al`: Class 0 mean = $2.04$ vs Class 1 mean = $2.03$ ($p = 0.915$, univariate ROC-AUC = $0.502$).
2. **Artificial Uniform Distributions**:
   - Continuous laboratory features follow rectangular/uniform distributions spanning arbitrary ranges (e.g., Blood Pressure $[70, 179]$, KS uniform test $p = 0.518$; Blood Urea $[10, 199]$, KS uniform test $p = 0.310$).
3. **Absence of Biological Covariance**:
   - The mean absolute pairwise correlation across all feature pairs is $r = 0.0253$, exactly matching the expected spurious correlation of uncorrelated random noise ($\sqrt{2 / (\pi \times 988)} = 0.0254$).
4. **Predictive Failure**:
   - A 5x10 repeated stratified cross-validation benchmark yields chance-level discrimination: Logistic Regression AUC = $0.5138 \pm 0.0326$, Random Forest AUC = $0.5283 \pm 0.0304$, XGBoost AUC = $0.5257 \pm 0.0319$.

## 5. Preserved Artifacts Inventory
- `ckd_raw.csv`: Original preserved raw data.
- `ckd.yaml`: Quarantined dataset configuration (preventing pipeline execution).
- `ckd_clean.parquet`: Quarantined preprocessed matrix.
- `ckd_comparators.parquet`: Benchmark comparator evaluations.
- `ckd_jitter.parquet`: Weight-jitter sensitivity evaluations.
- `negative_controls_ckd/`: Quarantined negative controls evaluations.
- `tables_ckd/`: TOST analysis and performance tables.
- `figures_ckd/`: Generated publication figures (F1-F5).
- `ckd_weights.json`: KDIGO 2024 expert weights profile.

## 6. Restoration Policy
CKD may only re-enter the active benchmark if the canonical 400-patient UCI dataset (`chronic_kidney_disease.arff`, Soundarapandian et al., 2015) is downloaded, preprocessed with proper in-fold imputation, and verified to exhibit genuine biomarker separation.
