# Data Card: Chronic Kidney Disease (CKD) [STATUS: QUARANTINED]

> [!CAUTION]
> **DATA INTEGRITY AUDIT / QUARANTINE NOTICE**:
> This dataset (`ckd_raw.csv`, $N=988$) originates from **BD-KDD: A Clinical Dataset of Kidney Disease Diagnosis** (Popular Diagnostic Center, Dhaka; Zenodo / BD-KDD), NOT the canonical 400-patient Apollo Hospitals / UCI CKD benchmark (Soundarapandian et al., 2015).
> **Data Integrity Sanity Test Failed**: In this matrix, the primary KDIGO clinical hallmarks show near-zero difference between Class 0 (control) and Class 1 (disease):
> - Serum Creatinine (`sc`): 7.41 mg/dL (Class 0) vs 7.65 mg/dL (Class 1) ($\Delta = +0.24$)
> - Urine Albumin (`al`): 2.04 (Class 0) vs 2.03 (Class 1) ($\Delta = -0.01$)
> As a consequence, machine learning models achieve near-random discrimination ($\approx 0.51$ ROC-AUC). All downstream experimental artifacts have been isolated into `data/quarantine/` and removed from active protocol synthesis.

## Dataset Overview
- **Dataset Name**: BD-KDD Kidney Disease Diagnosis Dataset (commonly misattributed as UCI CKD 988-row variant)
- **Source**: BD-KDD / Zenodo (Popular Diagnostic Center, Savar, Dhaka, Bangladesh)
- **Download URL**: https://zenodo.org/records/BD-KDD (`BD-KDD.csv`)
- **Raw File SHA-256**: `cb9d92f77f20859edaf73ba9236d8fed814e854016ab2eaf83e13ddba45bfaeb`
- **Quarantined Artifact Path**: `data/quarantine/ckd_raw.csv`, `data/quarantine/ckd_clean.parquet`
- **Target Variable**: `class` / `Class` (0 = non-CKD / control, 1 = CKD)
- **Sample Size**: 988 patient records ($N = 988$)
- **Number of Features**: 24 clinical, laboratory, and demographic features
- **Class Balance**: 48.7% control (0), 51.3% CKD (1)
- **Clinical Guideline Basis**: KDIGO (Kidney Disease: Improving Global Outcomes) 2012 / 2024 Clinical Practice Guidelines for CKD Evaluation and Management

---

## Features & Clinical Descriptions

| Feature | Description | Range / Categories | Type | KDIGO Role |
|---|---|---|---|---|
| `age` | Age in years | 20 – 79 | Continuous | Non-modifiable risk factor; eGFR declines with age |
| `bp` | Blood pressure (mm Hg) | 70 – 179 | Continuous | Hallmark cause and accelerator of renal parenchymal injury |
| `sg` | Urine specific gravity | 1.005 – 1.025 | Continuous / Ordinal | Reflects tubular concentrating capacity |
| `al` | Urine albumin | 0 – 4 | Ordinal | Core guideline hallmark for renal staging (CGA A1–A3) |
| `su` | Urine sugar | 0 – 4 | Ordinal | Indicator of glycemic control and tubular reabsorption |
| `rbc` | Red blood cells in urine | 0 = normal, 1 = abnormal | Binary | Active glomerular hematuria / nephritis |
| `pc` | Pus cells in urine | 0 = normal, 1 = abnormal | Binary | Urinary tract infection / interstitial nephritis |
| `pcc` | Pus cell clumps | 0 = not present, 1 = present | Binary | Severe pyelonephritis / inflammation |
| `ba` | Bacteria in urine | 0 = not present, 1 = present | Binary | Active bacterial infection |
| `bgr` | Blood glucose random (mg/dL) | 70 – 399 | Continuous | Diabetic nephropathy screening hallmark |
| `bu` | Blood urea (mg/dL) | 10 – 199 | Continuous | Nitrogenous waste clearance biomarker |
| `sc` | Serum creatinine (mg/dL) | 0.5 – 14.98 | Continuous | Core diagnostic hallmark for eGFR and G-staging |
| `sod` | Serum sodium (mEq/L) | 130 – 149 | Continuous | Fluid and electrolyte homeostasis |
| `pot` | Serum potassium (mEq/L) | 3.51 – 6.50 | Continuous | Electrolyte balance and fatal arrhythmia risk in CKD |
| `hemo` | Hemoglobin (g/dL) | 7.0 – 17.0 | Continuous | Anemia due to impaired renal erythropoietin secretion |
| `pcv` | Packed cell volume (%) | 20 – 54 | Continuous | Hematocrit monitoring for CKD anemia |
| `wbcc` | White blood cell count (/cu.mm) | 4012 – 14987 | Continuous | Non-specific inflammation marker |
| `rbcc` | Red blood cell count (millions/cu.mm) | 3.5 – 6.0 | Continuous | Hematologic index for renal anemia |
| `htn` | Hypertension history | 0 = no, 1 = yes | Binary | Primary etiology and progression accelerator |
| `dm` | Diabetes mellitus history | 0 = no, 1 = yes | Binary | Leading etiology of end-stage renal disease (ESRD) |
| `cad` | Coronary artery disease | 0 = no, 1 = yes | Binary | Major macrovascular comorbidity in CKD |
| `appet` | Appetite | 0 = poor, 1 = good | Binary | Symptom of uremic toxicity |
| `pe` | Pedal edema | 0 = no, 1 = yes | Binary | Hallmark sign of volume overload / nephrotic syndrome |
| `ane` | Anemia presentation | 0 = no, 1 = yes | Binary | Clinical presentation of deficient erythropoietin |

---

## Clinical Prioritization Weights
Clinical weights are external, expert-derived priors in $[0, 1]$ stored at `configs/weights/ckd_weights.json` and documented in `configs/weights/provenance.csv` grounded in KDIGO guidelines.

- **Primary Diagnostic Hallmarks ($w = 1.0$)**: `sc` (serum creatinine), `al` (albuminuria)
- **High Renal & Hemodynamic Indicators ($w = 0.8 - 0.9$)**: `bu` (0.9), `sg` (0.85), `htn` (0.85), `bp` (0.8), `dm` (0.8)
- **Secondary Complications & Hematology ($w = 0.65 - 0.75$)**: `hemo` (0.75), `pcv` (0.7), `bgr` (0.7), `ane` (0.7), `sod` (0.65), `pot` (0.65), `pe` (0.65)
- **Demographic & General Laboratory ($w = 0.4 - 0.55$)**: `age` (0.55), `su` (0.55), `rbc` (0.55), `rbcc` (0.55), `pc` (0.5), `cad` (0.45), `pcc` (0.4), `ba` (0.4), `wbcc` (0.4), `appet` (0.4)
