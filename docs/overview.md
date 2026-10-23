# DODA Methodology & Experimental Architecture Overview

This document provides a comprehensive reference of the experimental methodology, mathematical formulations, clinical cohorts, evaluation metrics, statistical testing protocols, and execution tracks for the DODA (Domain-Oriented Dual-stage Alignment) research framework.

---

## 1. The 4 Clinical Cohorts

DODA is evaluated across four clinical datasets spanning distinct data dimensions, clinical collection settings, and measurement modalities:

### 1.1 Diabetes Health Indicators
- **Dimensions**: $N = 10,000$ samples, $P = 21$ features.
- **Binary Target**: `Diabetes_binary` (0 = no diabetes, 1 = prediabetes / diabetes; ~14% positive prevalence).
- **Clinical Setting**: Population-level public health survey derived from the CDC Behavioral Risk Factor Surveillance System (BRFSS).
- **Guideline Basis**: World Health Organization (WHO) and American Diabetes Association (ADA) diagnostic standards. Prioritizes key lifestyle and metabolic hallmarks: Body Mass Index (`BMI`), High Blood Pressure (`HighBP`), High Cholesterol (`HighChol`), Physical Activity (`PhysActivity`), Age, and General Health (`GenHlth`).

### 1.2 Heart Disease (Cleveland Clinic)
- **Dimensions**: $N = 303$ samples, $P = 13$ features.
- **Binary Target**: `target` (presence of coronary heart disease $\ge 50\%$ diameter narrowing; ~54% prevalence).
- **Clinical Setting**: Clinical examination and non-invasive diagnostic testing matrix.
- **Guideline Basis**: American Heart Association (AHA) and American College of Cardiology (ACC) Angina Consensus criteria. Heavily weights classic ischemic signs: chest pain type (`cp`), exercise-induced ST depression (`oldpeak`), maximum heart rate achieved (`thalach`), and ST segment slope (`slope`).

### 1.3 Breast Cancer (Wisconsin Diagnostic - WDBC)
- **Dimensions**: $N = 569$ samples, $P = 30$ features.
- **Binary Target**: `diagnosis` (0 = benign, 1 = malignant; ~37% malignant prevalence).
- **Clinical Setting**: High-resolution digitized fine-needle aspirate (FNA) cytopathology image features.
- **Guideline Basis**: Clinical Cytopathology Guidelines & Nottingham Grading System standards. Weights nuclear pleomorphism, cell contour irregularity, clump thickness, and severe chromatin margin irregularities: concave points (`concave points_mean`), perimeter worst (`perimeter_worst`), radius worst (`radius_worst`), and texture worst (`texture_worst`).
- **Characteristics**: Highly collinear, continuous morphological measurements across mean, standard error, and extreme ("worst") nuclear values.

### 1.4 Chronic Kidney Disease (CKD)
- **Dimensions**: $N = 988$ samples, $P = 24$ features.
- **Binary Target**: `Class` (0 = no chronic kidney disease, 1 = chronic kidney disease).
- **Clinical Setting**: Inpatient hospital laboratory panel and bedside vitals record.
- **Guideline Basis**: KDIGO (Kidney Disease: Improving Global Outcomes) Clinical Practice Guidelines. Prioritizes diagnostic serum filtration markers: serum creatinine (`sc`), albuminuria (`al`), blood urea (`bu`), hemoglobin (`hemo`), and specific gravity (`sg`).
- **Characteristics**: Contains mixed numeric and categorical variables, missingness requiring median/mode imputation, and high laboratory noise.

---

## 2. The 5 Feature Selection (FS) Methods

DODA is selector-agnostic: it does not replace underlying feature selectors, but applies guideline steering on top of their raw continuous relevance scores. To test generalizability across distinct mathematical formulations, five foundational feature selection algorithms are evaluated:

### 2.1 ANOVA F-Test (Univariate Parametric Filter)
- **Mechanism**: Computes parametric variance ratios:
  $$F = \frac{\text{Between-Group Variance}}{\text{Within-Group Variance}}$$
- **Behavior**: Evaluates individual feature-label linear associations independently, ignoring pairwise collinearity or non-linear interactions. Fast and computationally cheap, serving as an unregularized baseline.

### 2.2 LASSO / L1-Penalized Logistic Regression (Embedded Linear)
- **Mechanism**: Solves an L1-regularized sparse convex optimization problem:
  $$\min_{\beta} -\ell(\beta) + \alpha \sum_{j=1}^P |\beta_j|$$
- **Behavior**: Simultaneously shrinks model coefficients and drives weak or collinear feature coefficients strictly to zero. Feature scores are extracted as standardized absolute coefficient magnitudes $|\beta_j|$.

### 2.3 Random Forest MDI (Embedded Non-linear Ensemble)
- **Mechanism**: Trains an ensemble of $100$ decorrelated decision trees and aggregates Mean Decrease in Impurity (Gini importance across splits).
- **Behavior**: Captures complex multi-way non-linear feature interactions and conditional dependencies across bootstrap samples without assuming linear separability.

### 2.4 mRMR (Minimum Redundancy Maximum Relevance - Information Theoretic Filter)
- **Mechanism**: Optimizes mutual information difference:
  $$\max_{S} \left[ \frac{1}{|S|} \sum_{i \in S} I(X_i; Y) - \frac{1}{|S|^2} \sum_{i,j \in S} I(X_i; X_j) \right]$$
- **Behavior**: Maximizes mutual information between candidate features and the clinical target while explicitly penalizing pairwise redundancy between candidate features.

### 2.5 Boruta (All-Relevant Tree-Ensemble Wrapper)
- **Mechanism**: Creates randomized shadow features by permuting existing feature columns, trains Random Forests, and performs two-sided statistical tests on whether real feature Z-scores exceed shadow max Z-scores across iterations.
- **Behavior**: Identifies all statistically relevant biological indicators rather than finding a minimal predictive subset.

---

## 3. Score Fusion Operators & Downstream Classifiers

DODA provides two alternative fusion operators to combine data-driven importance scores $S_{\text{norm}}(j) \in [0, 1]$ with guideline weights $w(j) \in [0, 1]$:

### 3.1 Hadamard Product (Multiplicative Scaling)
$$S_{\text{final}}(j) = S_{\text{norm}}(j) \cdot w(j)^\lambda$$
- Multiplies the continuous normalized importance score by the exponentiated clinical weight, where $\lambda \in [0.0, 0.25, 0.50, 0.75, 1.0]$.
- At $\lambda = 0.0$, the baseline data-driven ranking is strictly reproduced; at $\lambda = 1.0$, features lacking clinical backing ($w(j) \to 0$) are strongly suppressed.

### 3.2 Reciprocal Rank Fusion (RRF)
$$RRF(j) = \frac{1}{60 + r_{\text{math}}(j)} + \frac{\lambda}{60 + r_{\text{clin}}(j)}$$
- Fuses ordinal ranks rather than continuous values, mitigating score distribution disparities and calibration anomalies across selectors.

### 3.3 Downstream Classifiers Evaluated
- **Logistic Regression (LR)** ($C=1.0$, `max_iter=500`): Linear parametric benchmark.
- **Random Forest (RF)** ($100$ trees, `random_state=42`): Non-linear ensemble benchmark.
- **XGBoost (GBDT)** (`learning_rate=0.1`, `max_depth=4`, `eval_metric='logloss'`): Gradient-boosted decision trees (evaluated in negative controls and comparators).

---

## 4. Complete Metric Suite

All model evaluations record a multidimensional battery of metrics across three key dimensions:

### 4.1 Predictive Discrimination & Classification Performance
- **ROC-AUC (Primary Metric)**: Area under the Receiver Operating Characteristic curve. Measures rank discrimination invariant to decision threshold.
- **F1 Score**: Binary positive-class harmonic mean of precision and recall.
- **Macro F1 Score**: Unweighted mean of F1 scores across positive and negative classes.
- **PR-AUC (Average Precision)**: Area under the Precision-Recall curve, essential for evaluating class imbalance.
- **Balanced Accuracy**: Arithmetic mean of sensitivity and specificity.
- **Overall Accuracy**: Total correct classifications divided by total samples.

### 4.2 Clinical Alignment & Controllability
- **Spearman Rank Correlation ($\rho$)**: Nonparametric rank correlation between the fused feature ranking and the authentic guideline weights vector $w$. Validates whether increasing $\lambda$ monotonically aligns the feature subset with medical guidelines.
- **Guideline Hallmark Share**: Proportion of the selected Top-$K$ features that possess strong guideline backing ($w(j) \ge 0.8$).

### 4.3 Feature Selection Stability & Consistency
- **Top-$K$ Jaccard Identity to Baseline**: $\text{Jaccard} = \frac{|S_{\text{DODA}} \cap S_{\text{baseline}}|}{|S_{\text{DODA}} \cup S_{\text{baseline}}|}$. Quantifies feature set divergence induced by clinical steering.
- **Pairwise Cross-Fold Jaccard Stability**: Measures stability of selected feature subsets across the 50 cross-validation folds.

---

## 5. Statistical Testing & Rigor Protocol

To ensure conclusions are methodologically sound and guard against false positives, the following statistical protocols are enforced:

1. **$5 \times 10$ Repeated Stratified Cross-Validation**:
   - Every dataset is partitioned across 5 distinct random seeds, each undergoing 10-fold stratified cross-validation (50 independent folds per dataset).
   - All scaling, imputations, and feature selections are performed strictly in-fold to eliminate data leakage.
2. **Nadeau–Bengio Resampled $t$-Test Correction**:
   - Standard $t$-tests underestimate variance in cross-validation because training sets overlap. The standard error of paired differences $\Delta = \text{AUC}_{\text{DODA}} - \text{AUC}_{\text{base}}$ is corrected using:
     $$SE_{\text{corrected}} = \sqrt{\left(\frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) S^2}$$
     where $K=50$, $n_{\text{test}}/n_{\text{train}} = 1/9$.
3. **Formal Non-Inferiority Testing (RQ1)**:
   - DODA is evaluated against a pre-specified non-inferiority margin $\epsilon = 0.02$ (secondary $\epsilon = 0.01$).
   - Non-inferiority is formally established if the lower bound of the 95% one-sided confidence interval for $\Delta$ exceeds $-\epsilon$:
     $$\text{CI}_{\text{lower}} = \bar{\Delta} - t_{0.95, df} \cdot SE_{\text{corrected}} \ge -0.02$$
4. **Two One-Sided Tests (TOST) Equivalence Analysis**:
   - Tests whether DODA performance falls strictly within the equivalence bounds $[-\epsilon, +\epsilon]$, rejecting both underperformance and spurious divergence.
5. **Benjamini–Hochberg False Discovery Rate (FDR) Control**:
   - Across all selector, classifier, and fusion configurations, $p$-values are adjusted using the Benjamini–Hochberg procedure to control the false discovery rate at $q < 0.05$.
6. **Monotonic Rank Correlation Test**:
   - Assesses whether clinical alignment $\rho$ increases monotonically with respect to steering strength $\lambda$ across folds.

---

## 6. The 5 Dedicated Experimental Tracks

### Track 1: Primary Safety Non-Inferiority Grid (RQ1)
- **Objective**: Evaluates whether clinical weighting preserves predictive performance within non-inferiority margin $\epsilon = 0.02$ across 50 folds.
- **Grid**: Systematic sweep across all valid $K$ values, $\lambda \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$, selectors, and classifiers across 50 folds.

### Track 2: Clinical Controllability & Pareto Curve (RQ2)
- **Objective**: Evaluates the descriptive trade-off curve between clinical guideline alignment ($\rho$) and predictive accuracy (ROC-AUC).
- **Protocol**: Traces alignment growth across $\lambda$ sweeps and identifies operational default $\lambda^*$ via inner CV splits.

### Track 3: Negative Controls Specificity Grid (RQ3)
- **Objective**: Replaces authentic guideline weights with 4 control profiles to prove steering is driven by authentic medical evidence rather than random regularization:
  - *Permuted Weights*: Randomly shuffles the empirical guideline weights across features.
  - *Uniform Random Weights*: Assigns $w_j \sim \text{Uniform}(0, 1)$.
  - *Neutral Weights*: Assigns $w_j = 1.0$ (identically reproducing baseline selection to verify numerical consistency).
  - *Inverted Stress Weights*: Inverts guideline scores ($1.0 - w_j$) to deliberately steer toward clinically unsupported features.

### Track 4: Selector-Specific Knowledge Comparators (RQ3)
- **Objective**: Benchmarks DODA head-to-head against existing domain-guided methods at matched operational strength:
  - **DODA + LASSO** vs. **Penalty-Factor LASSO** (where feature penalty factors are inversely scaled: $\gamma_j = 1/w_j$).
  - **DODA + mRMR** vs. **Prior-Weighted mRMR** (where relevance is weighted: $I(X_j; Y) \cdot w_j$).

### Track 5: Weight-Jitter Sensitivity Analysis (Secondary)
- **Objective**: Assesses robustness against clinician subjectivity by injecting multiplicative uniform noise into guideline weights:
  $$w'_j = \text{clip}(w_j \cdot (1 + \delta_j), 0, 1), \quad \delta_j \sim \text{Uniform}(-\eta, \eta)$$
  for $\eta \in \{0.10, 0.20\}$ across 100 random draws per noise level, measuring output stability via Jaccard overlap and downstream ROC-AUC.

---

## 7. Experimental Matrix: Runs & Evaluations per Category

| Track | Protocol & Dimensions | Runs per Cohort | Total (4 Cohorts) |
|---|---|:---:|:---:|
| **1. Primary Cross-Validation Grid** | 50 folds $\times$ 3 selectors $\times$ 2 fusions $\times$ 5 $\lambda$ sweeps $\times$ 4 $K$ values $\times$ 2 classifiers | **12,000 runs** | **48,000 runs** |
| **2. Negative Controls Grid (RQ3)** | 50 folds $\times$ 4 selectors $\times$ 5 control types (True, Neutral, Permuted, Random, Inverted) $\times$ 4 $K$ values $\times$ 3 classifiers | **12,000 runs** | **48,000 runs** |
| **3. Selector-Specific Comparators (RQ3)** | 50 folds $\times$ 4 methods (PF-LASSO, DODA-LASSO, Prior-mRMR, DODA-mRMR) $\times$ 4 $K$ values $\times$ 3 classifiers | **2,400 runs** | **9,600 runs** |
| **4. Weight-Jitter Sensitivity** | 100 random draws $\times$ 2 noise levels ($\pm 10\%, \pm 20\%$) $\times$ 4 selectors $\times$ 4 $K$ values | **3,200 evaluations** | **12,800 evaluations** |
| **Total Empirical Evaluations** | | **29,600 per cohort** | **$\approx$ 118,400 evaluations** |
