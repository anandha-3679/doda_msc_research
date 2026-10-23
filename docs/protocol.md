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
- **Controllability**: Alignment between selected features and clinical priority profiles is measured via Spearman rank correlation $\rho$ and Top-$K$ clinical weight share across $\lambda \in [0.0, 0.25, 0.50, 0.75, 1.0]$.
- **Descriptive Trade-off**: Characterize the alignment-vs-performance curve ($\rho$ vs. $\Delta \text{ROC-AUC}$) as a descriptive Pareto frontier.
- **Operational $\lambda^*$ Selection**: Any recommended $\lambda^*$ must be selected via inner nested CV on the training folds (the largest $\lambda$ satisfying non-inferiority on train validation splits), never by inspecting test fold outcomes.

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
- **CKD Baseline Check ($N=988$, $D=24$)**: Compute baseline AUC before classifying as a "hard" benchmark.

---

## 4. Execution Order & Storage Rules
1. Order: Heart Disease $\rightarrow$ WDBC $\rightarrow$ Diabetes.
2. Grid definitions are strictly frozen prior to running experiments.
3. Results are append-only; existing runs in `results/*/runs.parquet` are never overwritten.
4. Tables and figures are generated strictly via `scripts/make_tables.py` and `scripts/make_figures.py`.
