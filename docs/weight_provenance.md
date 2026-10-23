# Clinical Weight Provenance & Evidence Grading

## 1. Principles of Weight Specification in DODA
DODA uses external, versioned JSON weight files in `configs/weights/<dataset>_weights.json` to inject domain priors into feature selection. Every feature weight $w_j \in [0.0, 1.0]$ represents an explicit clinical or pathophysiological priority:
- $w = 1.0$: Pathognomonic symptoms, primary diagnostic criteria, established major independent biomarkers.
- $w = 0.8 - 0.9$: Major clinical guidelines (AHA, ADA, WHO, KDIGO) identifying established secondary risk factors or hallmark structural changes.
- $w = 0.5 - 0.7$: Contributing demographic covariates, secondary functional measurements, or general health indicators.
- $w = 0.2 - 0.4$: Non-specific lifestyle indicators, secondary socio-economic proxies, or intra-sample measurement noise.

> **Design Choice Notice**: Mapping qualitative clinical consensus into discrete numeric weights in $[0, 1]$ is a design choice. To test the robustness of this operationalization, DODA conducts systematic **weight-jitter sensitivity analyses** ($\pm 10\%$ and $\pm 20\%$ perturbation across 100 draws), verifying that rank ordering and downstream performance are not fragile to minor changes in weight values.

---

## 2. Guideline Provenance by Dataset

### 2.1 CDC BRFSS Diabetes Health Indicators
- **Overall Category**: Guideline-derived (WHO / American Diabetes Association [ADA] Standards of Care).
- **Primary Biomarkers ($w=1.0$)**:
  - `high_bp`, `high_chol`, `bmi`, `heart_diseaseor_attack`, `stroke`: Established metabolic syndrome and vascular risk factors (ADA Standards of Medical Care in Diabetes, 2023; WHO Global Report on Diabetes).
- **Functional Burden ($w=0.8 - 0.9$)**:
  - `phys_hlth`, `diff_walk`: Functional morbidity indicators reflecting vascular decline and neuropathy.
- **Demographic & Lifestyle Proxies ($w=0.2 - 0.4$)**:
  - `income`, `education`, `sex`, `fruits`, `veggies`: Socio-economic proxies and indirect lifestyle covariates.

### 2.2 UCI Cleveland Heart Disease
- **Overall Category**: Guideline-derived (American College of Cardiology [ACC] / American Heart Association [AHA] Guidelines).
- **Core Pathognomonic & Angiographic Hallmarks ($w=1.0$)**:
  - `cp`: Angina symptom categorization (AHA/ACC Chest Pain Guidelines).
  - `exang`: Exercise-induced ischemia hallmark.
  - `oldpeak`: Quantitative ST depression under stress.
  - `ca`: Number of major vessels with fluoroscopic narrowing (direct anatomical severity).
  - `thal`: Thallium scintigraphy defect status (fixed vs reversible ischemia).
- **Hemodynamic & Physiological Predictors ($w=0.8 - 0.9$)**:
  - `trestbps`: Resting blood pressure (AHA Hypertension guidelines).
  - `chol`: Total serum cholesterol (NCEP ATP III / AHA).
  - `thalach`: Peak functional exercise heart rate.
  - `restecg`: Resting ST-T wave abnormalities.
- **Secondary Modifiers ($w=0.6$)**:
  - `sex`, `fbs`: Fasting blood sugar (comorbidity indicator) and demographic risk.

### 2.3 Wisconsin Diagnostic Breast Cancer (WDBC)
- **Overall Category**: Quantitative Cytopathology Consensus (Wolberg, Street, Mangasarian 1995 / FNA Cytology Standards).
- **Evidence Level Distinction**: Unlike Diabetes and Heart Disease, which rest on broad national clinical practice guidelines (ADA, AHA), WDBC weights are derived from **quantitative nuclear morphometry from digitized Fine Needle Aspirate (FNA) images**.
- **Peak Nuclear Atypia Hallmarks ($w=1.0$)**:
  - `worst_radius`, `worst_perimeter`, `worst_area`, `worst_concavity`, `worst_concave_points`: Extreme nuclear enlargement and deep irregular indentations diagnostic of invasive carcinoma.
  - *Note*: Weakest formal guideline backing; graded as cytopathological consensus / image-derived nuclear atypia.
- **Intra-Sample Errors / Heterogeneity ($w=0.2 - 0.5$)**:
  - `*_error`: Standard errors representing variation across nuclei on a single slide.
- **Border Roughness ($w=0.3$)**:
  - `*_fractal_dimension`: Nuclear perimeter coastline approximation.
