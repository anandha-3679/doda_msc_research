# DODA Experimental Protocol (Revised)

## 1. Primary Objective & Scope
DODA (Decoupled Operator-Driven Architecture) is a selector-agnostic integration layer designed to inject an external, versioned, and auditable clinical-weight file between any feature selector's normalized scores and final Top-$K$ ranking. 

**Core Claim**: A domain expert or clinician can steer feature priorities toward guideline-concordant markers without hurting predictive performance beyond a pre-declared statistical margin. DODA does NOT claim that clinical weights improve predictive accuracy.

---

## 2. Research Questions & Statistical Hypotheses

### Primary: RQ1 (Performance Non-Inferiority)
- **Primary Hypothesis**: DODA-selected feature subsets are non-inferior to baseline purely data-driven feature subsets in discrimination (ROC-AUC) and balance (F1).
  $$\Delta = \text{metric}_{\text{DODA}} - \text{metric}_{\text{baseline}}$$
  - **Decision Rule**: The lower bound of the one-sided $95\%$ confidence interval (equivalent to the lower limit of a $90\%$ two-sided CI) of $\Delta$ must satisfy:
    $$\text{CI}_{\text{lower}} \ge -\epsilon$$
  - **Margins**: $\epsilon = 0.02$ (primary safety margin); $\epsilon = 0.01$ (secondary robustness check, noted as underpowered on small cohorts like $N \approx 300$).
  - **Variance Correction**: Nadeau–Bengio resampled cross-validation correction applied across all $5 \times 10$ folds:
    $$\sigma^2_{\text{corrected}} = \left(\frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s^2_{\Delta}$$
  - **Secondary Evaluation**: Two-sided TOST (Two One-Sided Tests) for practical equivalence ($|\Delta| < \epsilon$). If DODA exceeds $+\epsilon$, it is reported as *"non-inferior and higher on this cohort"*, without attributing the elevation to clinical weights.
  - **Multiple Testing**: Benjamini–Hochberg False Discovery Rate (FDR) control applied across all compared selector $\times$ classifier $\times$ Top-$K$ combinations within each dataset.

### RQ2 (Ranking Controllability & Descriptive Trade-off)
- **Controllability**: Alignment between selected features and clinical priority profiles is measured via Spearman rank correlation $\rho$ and Top-$K$ clinical weight share across a prespecified fixed descriptive grid $\lambda \in [0.0, 0.25, 0.50, 0.75, 1.0]$.
- **Descriptive Trade-off**: Characterize the alignment-vs-performance curve ($\rho$ vs. $\Delta \text{ROC-AUC}$) as an empirical Pareto frontier.
- **Fixed-Grid Design & Deployment $\lambda^*$ Workflow**: Current experimental results report the complete prespecified descriptive grid across all folds to map the full empirical trade-off. Dynamic selection of a single operational default $\lambda^*$ via inner nested cross-validation on training splits (selecting the maximal $\lambda$ satisfying train validation non-inferiority) represents an intended future clinical deployment workflow, not an implemented experimental result in this study.

### RQ3 (Negative Controls & Benchmark Comparators)
1. **Negative Controls Grid (Frozen across all datasets)**:
   - Evaluated at $\lambda = 1.0$, primary fusion method (Hadamard), main $K$ values, all 5 selectors (ANOVA, LASSO, mRMR, Random Forest, Boruta), and all 3 classifiers (Logistic Regression, Random Forest, XGBoost).
   - **Permuted / Shuffled Weights (Main Control)**: Randomly permute the real weight vector across feature indices, preserving the exact empirical weight multiset and distribution.
   - **Random Weights**: Uniform random draws $w_j \sim \mathcal{U}(0, 1)$.
   - **Neutral Weights**: All weights identically set to $1.0$ (sanity check: must reproduce baseline rankings under Hadamard).
   - **Inverted Weights ($w \rightarrow 1 - w$)**: Extreme stress test. Suppresses guideline hallmarks; severe degradation expected under Hadamard and weakly informative.
   - *Hypothesis rule*: No assumed outcome. Real guideline weights are justified if they achieve superior clinical alignment ($\rho$, Top-$K$ guideline share) at equivalent performance.

2. **Benchmark Comparators (Selector-Specific)**:
   - **Penalty-Factor LASSO**: Feature-specific penalties $\lambda_j = \lambda_0 / w_j$ (or column rescaling $X_j \cdot w_j$). Compared directly against DODA + LASSO.
   - **Prior-Weighted mRMR**: Mutual information quotient modified by clinical priors $I(X_j; Y) \cdot w_j / \frac{1}{|S|} \sum I(X_j; X_s)$. Compared directly against DODA + mRMR.

3. **Weight-Jitter Sensitivity Analysis**:
   - Multiplicative perturbation: $w_j' = \text{clip}(w_j \cdot (1 + \delta), 0, 1)$, where $\delta \sim \mathcal{U}(-\eta, +\eta)$ for $\eta \in \{0.10, 0.20\}$.
   - 100 draws per setting with fixed reproducible seeds.
   - Primary metric: Top-$K$ Jaccard stability vs. unjittered ranking across all selectors.
   - Performance metric: Evaluated with fast Logistic Regression. Boruta omitted from jitter draws due to quadratic compute overhead.

---

## 3. Dataset-Specific Protocol Rules & Ceiling Handling

- **UCI Cleveland Heart Disease ($N=303$, $D=13$)**: First evaluation cohort for controls and comparators. Small $N$ requires explicit reporting of finite-sample variance and wider Nadeau–Bengio CIs.
- **Wisconsin Diagnostic Breast Cancer ($N=569$, $D=30$)**: 
  - Baseline AUC is near ceiling ($\approx 0.99$), rendering equivalence at large $K$ weakly informative.
  - **Small-$K$ Track**: Evaluate $K \in \{2, 3\}$ for both baseline and DODA under paired 5x10 CV; reuse $K=5$ from the main grid.
  - Stored separately in `results/breast_cancer_small_k/runs.parquet`.
- **CDC BRFSS Diabetes ($N=10,000$ stratified subsample from $253,680$, $D=21$, seed 42)**: Large population epidemiological cohort.
- **Canonical UCI CKD ($N=400$, $D=24$)**: Apollo Hospitals benchmark evaluating KDIGO priors under paired 5x10 CV across all K budgets ($K \in [5, 10, 15, 20]$). High baseline separation acts as a clinical alignment and invariance testbed. (Suspect BD-KDD 988-row variant quarantined into `data/quarantine/`).

---

## 4. Primary Grid Definition Across All Four Datasets

The primary benchmark grid is evaluated uniformly across all **4 eligible clinical datasets** (Heart Disease, WDBC Breast Cancer, Canonical UCI CKD, and CDC BRFSS Diabetes) with the following mandatory design:
- **5 Selectors**: ANOVA, LASSO, Random Forest, mRMR, and Boruta.
- **3 Classifiers / Models**: Logistic Regression (`logreg`), Random Forest (`random_forest`), and XGBoost (`xgboost`).
- **4 Top-$K$ Budgets**: 4 dataset-appropriate Top-$K$ values ($[3, 5, 8, 10]$ for Heart Disease; $[5, 10, 15, 20]$ for WDBC, CKD, Diabetes).
- **5 Lambdas**: Prespecified fixed descriptive grid $\lambda \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$.
- **2 Fusion Operators**: Hadamard and RRF.
- **Cross-Validation**: 5 splits $\times$ 10 repeats = 50 folds Repeated Stratified K-Fold CV.
- Total evaluations per dataset = $50 \times 5 \times 2 \times 5 \times 4 \times 3 = 30,000$ evaluations ($120,000$ total evaluations across the 4 cohorts).

---

## 5. Execution Order & Storage Rules
1. Order: Heart Disease $\rightarrow$ WDBC $\rightarrow$ Canonical UCI CKD $\rightarrow$ Diabetes.
2. Grid definitions are strictly frozen prior to running experiments.
3. Results are saved in isolated runs directories (`results/<dataset>/runs.parquet`). Historical baseline runs are preserved in versioned backups.
4. Tables and figures are generated strictly via `scripts/make_tables.py` and `scripts/make_figures.py`.
