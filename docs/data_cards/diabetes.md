# Dataset Card: CDC BRFSS Diabetes Health Indicators

## Overview
- **Dataset Name**: CDC Behavioral Risk Factor Surveillance System (BRFSS) - Diabetes Health Indicators
- **Task**: Binary Classification (Diabetes or Prediabetes vs. No Diabetes)
- **Primary Source**: Centers for Disease Control and Prevention (CDC) BRFSS Survey
- **Samples (N)**: 253,680 total population rows; evaluated via a stratified subsample of $N=10,000$ (seed $42$) for cross-validation efficiency
- **Features (D)**: 21 predictive features + 1 binary target
- **Target Variable**: `diabetes_binary` (0: No diabetes, 1: Diabetes or Prediabetes)
- **Class Balance**: 86.07% negative (0), 13.93% positive (1) (Class imbalance ratio: ~6.18:1)
- **Baseline Performance**: ROC-AUC $\approx 0.7917$ ($\lambda=0.0$)
- **Artifact Path**: `data/processed/diabetes_clean.parquet`

## Feature Dictionary

| Feature Name | Type | Domain / Values | Description |
|---|---|---|---|
| `high_bp` | Binary | {0, 1} | High blood pressure diagnosed |
| `high_chol` | Binary | {0, 1} | High blood cholesterol diagnosed |
| `chol_check` | Binary | {0, 1} | Cholesterol check within past 5 years |
| `bmi` | Continuous | [12, 98] | Body Mass Index (kg/m²) |
| `smoker` | Binary | {0, 1} | Smoked at least 100 cigarettes in lifetime |
| `stroke` | Binary | {0, 1} | History of stroke |
| `heart_diseaseor_attack`| Binary | {0, 1} | Coronary heart disease or myocardial infarction |
| `phys_activity` | Binary | {0, 1} | Physical activity in past 30 days |
| `fruits` | Binary | {0, 1} | Consumes fruit 1+ times per day |
| `veggies` | Binary | {0, 1} | Consumes vegetables 1+ times per day |
| `hvy_alcohol_consump` | Binary | {0, 1} | Heavy alcohol consumption (men >14 drinks/wk, women >7) |
| `any_healthcare` | Binary | {0, 1} | Has health care coverage |
| `no_docbc_cost` | Binary | {0, 1} | Needed doctor in past 12 mos but could not due to cost |
| `gen_hlth` | Ordinal | 1-5 | Self-reported general health (1: Excellent, 5: Poor) |
| `ment_hlth` | Integer | 0-30 | Days of poor mental health in past 30 days |
| `phys_hlth` | Integer | 0-30 | Days of poor physical health in past 30 days |
| `diff_walk` | Binary | {0, 1} | Serious difficulty walking or climbing stairs |
| `sex` | Binary | {0, 1} | Female (0) / Male (1) |
| `age` | Ordinal | 1-13 | 5-year age categories (1: 18-24, 13: 80+) |
| `education` | Ordinal | 1-6 | Education level attained (1: Never attended, 6: College 4+ yrs) |
| `income` | Ordinal | 1-8 | Annual household income bracket (1: <$10k, 8: >$75k) |

## Preprocessing Protocol
- Column names transformed into standardized `snake_case`.
- Checked for null values; missing values dropped if found.
- All in-fold imputation, feature normalization, and scaling must be fit strictly on training splits.
