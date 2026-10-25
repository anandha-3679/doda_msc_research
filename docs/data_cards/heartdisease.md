# Data Card: UCI Cleveland Heart Disease

## Dataset Overview
- **Dataset Name**: UCI Heart Disease (Cleveland Database)
- **Source**: UCI Machine Learning Repository / Cleveland Clinic Foundation (Detrano et al., 1989)
- **Download URL**: https://archive.ics.uci.edu/static/public/45/heart+disease.zip (`processed.cleveland.data`)
- **Raw File SHA-256**: `13f2c7cc1ebfc284ead179a49d04ef4935ede24cff35452bf19a8eb643c7a8af`
- **Target Variable**: `target` (0 = no presence/diameter narrowing < 50%, 1 = presence/diameter narrowing $\ge$ 50%)
- **Sample Size**: 303 patient records (note: small $N$ yields wider confidence intervals under Nadeau–Bengio correction; $\epsilon=0.01$ is underpowered)
- **Number of Features**: 13 clinical, hemodynamic, and demographic features
- **Class Balance**: 54.1% negative (0), 45.9% positive (1)
- **Baseline Performance**: ROC-AUC $\approx 0.8699$ ($\lambda=0.0$)
- **Biomarker Sanity Verification**: Heart disease class (1) shows substantial elevations in pathognomonic angiographic hallmarks `ca` (mean 1.14 vs 0.27 vessels), `oldpeak` (1.57 vs 0.59 mm ST depression), and typical angina `cp` (3.59 vs 2.79). Verified by `tests/test_dataset_sanity.py`.

---

## Features & Clinical Descriptions

| Feature | Description | Range / Categories | Type |
|---|---|---|---|
| `age` | Age in years | 29 – 77 | Continuous |
| `sex` | Biological sex | 1 = male, 0 = female | Binary |
| `cp` | Chest pain type | 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic | Categorical / Ordinal |
| `trestbps` | Resting blood pressure (mm Hg on admission) | 94 – 200 | Continuous |
| `chol` | Serum cholesterol (mg/dl) | 126 – 564 | Continuous |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 = true, 0 = false | Binary |
| `restecg` | Resting electrocardiographic results | 0: normal, 1: ST-T wave abnormality, 2: probable/definite LV hypertrophy | Categorical |
| `thalach` | Maximum heart rate achieved during exercise | 71 – 202 | Continuous |
| `exang` | Exercise induced angina | 1 = yes, 0 = no | Binary |
| `oldpeak` | ST depression induced by exercise relative to rest | 0.0 – 6.2 | Continuous |
| `slope` | Slope of peak exercise ST segment | 1: upsloping, 2: flat, 3: downsloping | Categorical / Ordinal |
| `ca` | Number of major vessels colored by flourosopy | 0 – 3 (4 missing values imputed in-fold) | Discrete |
| `thal` | Thallium scintigraphy heart defect status | 3 = normal, 6 = fixed defect, 7 = reversible defect (2 missing values imputed in-fold) | Categorical |

---

## Clinical Prioritization Weights
Clinical weights are external, expert-derived and guideline-backed priors in $[0, 1]$ stored at `configs/weights/heartdisease_weights.json` and documented in `configs/weights/provenance.csv`.

- **Core Pathognomonic & Angiographic Markers ($w = 1.0$)**: `cp`, `exang`, `oldpeak`, `ca`, `thal`
- **Strong Hemodynamic & Physical Markers ($w = 0.9$)**: `age`, `trestbps`, `slope`
- **Metabolic & Functional Biomarkers ($w = 0.8$)**: `chol`, `restecg`, `thalach`
- **Secondary Modifiers ($w = 0.6$)**: `sex`, `fbs`
