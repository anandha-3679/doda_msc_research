# Data Card: Wisconsin Diagnostic Breast Cancer (WDBC)

## Dataset Overview
- **Dataset Name**: Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source**: UCI Machine Learning Repository / University of Wisconsin Hospitals, Madison (Wolberg, Street, Mangasarian, 1995)
- **Download URL**: https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip (`wdbc.data`)
- **Raw File SHA-256**: `11218503215d46bf9eb4e9bcf7da97b878f0cdae46c75863d00e5009ab91a76c`
- **Target Variable**: `diagnosis` (0 = Benign, 1 = Malignant)
- **Sample Size**: 569 fine-needle aspirate (FNA) patient records
- **Number of Features**: 30 quantitative nuclear morphology attributes (10 base morphological features $\times$ mean, standard error, and worst/largest)
- **Class Balance**: 62.7% Benign ($n=357$), 37.3% Malignant ($n=212$)
- **Baseline Performance**: ROC-AUC $\approx 0.9894$ ($\lambda=0.0$). **Ceiling Effect Warning**: Because baseline discrimination approaches $0.99$, performance equivalence at $K \ge 5$ is weakly informative; extreme sparsity tracks ($K \in \{2, 3\}$) are evaluated separately to test whether DODA preserves discrimination under severe feature constraints.
- **Missing Values**: 0 (Complete dataset)
- **Biomarker Sanity Verification**: Malignant tumors (1) exhibit extreme enlargement and atypia across `worst_concave_points` (0.182 vs 0.074, $>2.4\times$), `worst_perimeter` (141.4 vs 87.0 mm), and `mean_radius` (17.46 vs 12.15 mm). Verified by `tests/test_dataset_sanity.py`.

---

## Features & Cytopathological Descriptions

Ten real-valued features describing characteristics of the cell nuclei present in digitized images of fine needle aspirates:
1. `radius`: Mean of distances from center to points on the perimeter
2. `texture`: Standard deviation of gray-scale values
3. `perimeter`: Total nuclear boundary length
4. `area`: Total nuclear cross-sectional area
5. `smoothness`: Local variation in radius lengths
6. `compactness`: $\frac{\text{perimeter}^2}{\text{area}} - 1.0$
7. `concavity`: Severity of concave portions of the nuclear contour
8. `concave_points`: Number of concave indentations of the contour
9. `symmetry`: Relative nuclear symmetry
10. `fractal_dimension`: Nuclear boundary roughness ("coastline approximation" $- 1$)

Each nuclear feature is computed across three distributions per slide:
- `mean_*` (10 features): Average value across cell nuclei in the sample.
- `*_error` (10 features): Standard error of the mean, reflecting intra-tumor cellular heterogeneity.
- `worst_*` (10 features): Mean of the three largest values, capturing the most atypical/malignant cells.

---

## Clinical Prioritization Weights
Clinical weights are domain-informed cytopathological priors in $[0, 1]$ stored at `configs/weights/breast_cancer_weights.json` and documented in `configs/weights/provenance.csv`.

- **Peak Nuclear Atypia & Malignancy Hallmarks ($w = 1.0$)**:
  - `worst_radius`, `worst_perimeter`, `worst_area`: Marked nuclear enlargement (macronucleoli, pleomorphism).
  - `worst_concavity`, `worst_concave_points`: Deep nuclear indentations and irregular contours characteristic of invasive carcinoma.
- **Core Morphological Baseline Markers ($w = 0.9$)**:
  - `mean_radius`, `mean_perimeter`, `mean_area`: Gross nuclear enlargement.
  - `mean_concavity`, `mean_concave_points`: Frequent nuclear clefting and grooving.
- **Structural Irregularity ($w = 0.7 - 0.8$)**:
  - `worst_compactness` ($0.8$), `mean_compactness` ($0.7$): Distorted cell shape.
- **Secondary Nuclear Texture & Polarity ($w = 0.4 - 0.6$)**:
  - `worst_texture` ($0.6$), `worst_symmetry` ($0.6$), `worst_smoothness` ($0.5$), `mean_texture` ($0.5$), `mean_symmetry` ($0.5$), `mean_smoothness` ($0.4$).
- **Variance / Intra-Sample Error Features ($w = 0.2 - 0.5$)**:
  - `radius_error`, `perimeter_error`, `area_error` ($0.5$).
  - Minor texture/smoothness/symmetry errors ($0.2 - 0.4$).
- **Border Roughness ($w = 0.3$)**:
  - `mean_fractal_dimension`, `worst_fractal_dimension` ($0.3$), `fractal_dimension_error` ($0.2$).
