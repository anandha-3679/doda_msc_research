# Data Card: Canonical UCI Chronic Kidney Disease (CKD)

## 1. Overview
- **Dataset Name**: Early Stage Indian Chronic Kidney Disease (CKD) Dataset
- **Clinical Setting**: Apollo Hospitals, Managiri, Karaikudi, Tamil Nadu, India
- **Source**: UCI Machine Learning Repository
- **Creators / Collectors**: Dr. P. Soundarapandian (Apollo Hospitals), L. Jerlin Rubini, Dr. P. Eswaran (Alagappa University)
- **Primary Pathology**: Chronic Kidney Disease (CKD) diagnosis staging based on clinical and laboratory biomarkers
- **Number of Instances**: 400 patients (250 CKD [62.5%], 150 non-CKD / healthy control [37.5%])
- **Number of Attributes**: 24 predictors (11 continuous numerical, 13 categorical nominal/ordinal) + binary class label (`class`)

---

## 2. Distinction from Quarantined BD-KDD Dataset
- **Quarantined BD-KDD Artifact** ($N=988$): Provenance traced to Savar, Bangladesh. Quarantined in `data/quarantine/` due to zero diagnostic separation in core biomarkers (`sc` $\approx 7.4$ vs $7.6$, `al` $\approx 2.03$ vs $2.04$).
- **Canonical UCI Benchmark** ($N=400$): Demonstrates hallmark physiological differences:
  - Serum Creatinine (`sc`): CKD mean = $4.41 \text{ mg/dL}$ vs Control mean = $0.87 \text{ mg/dL}$ ($\Delta = +3.54 \text{ mg/dL}$)
  - Urine Albumin (`al`): CKD mean = $1.72$ vs Control mean = $0.00$
  - Hemoglobin (`hemo`): CKD mean = $10.65 \text{ g/dL}$ vs Control mean = $15.19 \text{ g/dL}$
  - Random Blood Glucose (`bgr`): CKD mean = $175.4 \text{ mg/dL}$ vs Control mean = $107.7 \text{ mg/dL}$

---

## 3. Features & Clinical Weighting (KDIGO Guidelines)
Features are weighted in `configs/weights/ckd_weights.json` based on the KDIGO (Kidney Disease: Improving Global Outcomes) 2012/2024 Practice Guidelines:

| Feature | Description | Unit / Scale | Weight | Clinical Guideline Rationale |
|---|---|---|---|---|
| `sc` | Serum Creatinine | mg/dL | 1.00 | Core diagnostic biomarker for eGFR calculation and G-staging |
| `al` | Urine Albumin | 0–5 | 1.00 | CGA albuminuria staging (A1–A3) |
| `bu` | Blood Urea Nitrogen | mg/dL | 0.90 | Uremic retention marker |
| `bp` | Blood Pressure | mmHg | 0.90 | Major cause and progression accelerator |
| `sg` | Specific Gravity | 1.005–1.025 | 0.90 | Tubular concentrating ability |
| `bgr` | Blood Glucose Random | mg/dL | 0.90 | Diabetic kidney disease risk factor |
| `htn` | Hypertension | 0/1 | 0.90 | Major etiology and vascular damage factor |
| `dm` | Diabetes Mellitus | 0/1 | 0.90 | Leading global etiology of ESRD |
| `pc` | Pus Cell | 0/1 | 0.90 | Pyuria / interstitial inflammation |
| `rbc` | Red Blood Cells | 0/1 | 0.80 | Glomerular hematuria indicator |
| `hemo` | Hemoglobin | g/dL | 0.80 | Renal erythropoietin deficiency anemia |
| `pcv` | Packed Cell Volume | % | 0.80 | Hematocrit tracking anemia severity |
| `pe` | Pedal Edema | 0/1 | 0.80 | Fluid overload / nephrotic syndrome |
| `age` | Age | years | 0.70 | Age-related eGFR decline |
| `sod` | Serum Sodium | mEq/L | 0.70 | Impaired water excretion |
| `pot` | Serum Potassium | mEq/L | 0.70 | Hyperkalemia complication |
| `rbcc` | Red Blood Cell Count | M/cmm | 0.70 | Erythropoietic suppression |
| `ane` | Clinical Anemia | 0/1 | 0.70 | Symptomatic chronic renal dysfunction |
| `pcc` | Pus Cell Clumps | 0/1 | 0.70 | Severe UTI / interstitial nephritis |
| `su` | Urine Sugar | 0–5 | 0.60 | Glycosuria / proximal tubular damage |
| `cad` | Coronary Artery Disease | 0/1 | 0.60 | Cardiovascular comorbidity |
| `wbcc` | White Blood Cell Count | cells/cumm | 0.50 | Systemic inflammation marker |
| `ba` | Bacteria | 0/1 | 0.50 | Infection confounding status |
| `appet` | Appetite | 0/1 | 0.50 | Uremic anorexia symptom |

---

## 4. Preprocessing & Artifact Paths
- **Raw File**: `data/raw/Chronic_Kidney_Disease/chronic_kidney_disease_full.arff`
- **Clean Parquet**: `data/processed/ckd_uci_clean.parquet`
- **Config**: `configs/datasets/ckd_uci.yaml`
- **Evaluation Output**: `results/ckd_uci/`
