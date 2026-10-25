# Protocol Changelog

This changelog formally records all adjustments, additions, and reframings to the experimental design, noting the effective date, rationale, and whether the modification occurred post-hoc (after seeing preliminary results).

---

### [2026-10-22] - Revision 1.1: Non-Inferiority Reframing & Protocol Refinements

- **Change 1: Primary Hypothesis Reframed to One-Sided Non-Inferiority (Post-hoc)**
  - *Description*: Primary RQ1 criterion is now one-sided non-inferiority: lower bound of $95\%$ CI of $\Delta \text{ROC-AUC} \ge -\epsilon$ ($\epsilon=0.02$ primary, $0.01$ secondary) with Nadeau–Bengio resampled CV correction. Two-sided TOST equivalence is preserved as secondary.
  - *Post-hoc status*: **Yes**. Applied after viewing initial equivalence results on Diabetes and Heart Disease to better align with clinical safety evaluation standards ("do no harm").
  - *Data Integrity*: All prior fold-level metrics in `results/*/runs.parquet` remain untouched; statistics are recomputed without model retraining.

- **Change 2: Negative Controls Specification (Pre-declared for execution)**
  - *Description*: Added permuted/shuffled weights (main control preserving weight multiset), uniform random weights, neutral weights (all 1.0), and inverted weights ($1 - w$, extreme stress test) evaluated at $\lambda=1.0$, primary fusion method (Hadamard), all 5 selectors, and all 3 classifiers.
  - *Rationale*: Address reviewer critique regarding whether DODA selection is sensitive to specific guideline knowledge or merely any weight vector.

- **Change 3: Weight-Jitter Sensitivity Analysis (Pre-declared for execution)**
  - *Description*: Added multiplicative perturbation of weights by $\pm 10\%$ and $\pm 20\%$ (clipped to $[0, 1]$, 100 draws, fixed seeds) evaluating Top-$K$ Jaccard stability and Logistic Regression performance. Boruta excluded due to compute intensity.
  - *Rationale*: Test sensitivity to semi-quantitative weight definitions.

- **Change 4: Benchmark Comparators Definition (Pre-declared for execution)**
  - *Description*: Added Penalty-Factor LASSO ($penalty = 1/w$) vs DODA+LASSO, and Prior-Weighted mRMR vs DODA+mRMR.
  - *Rationale*: Compare against selector-specific knowledge injection baselines.

- **Change 5: WDBC Small-$K$ Track ($K \in \{2, 3\}$) (Post-hoc)**
  - *Description*: Added dedicated runs for $K \in \{2, 3\}$ on WDBC with paired baseline and DODA tracks, saved to `results/breast_cancer_small_k/runs.parquet`. Reuses $K=5$ from main grid.
  - *Post-hoc status*: **Yes**. Driven by the observation that WDBC baseline AUC is $\approx 0.99$ at $K \ge 5$ (ceiling effect), requiring evaluation under extreme feature compression to test whether DODA preserves discrimination under severe sparsity.

- **Change 6: Terminology and Positioning Precision**
  - *Description*: Standardized positioning wording: DODA is "selector-agnostic" (not "model-agnostic"). Prohibited claims of "proven zero-cost" or "enhanced stability". Weight provenance clarified as guideline-derived (WHO/ADA, AHA/ACC, KDIGO) with Breast Cancer (WDBC) weights author-assigned from FNA diagnostic literature.

  - *Rationale*: Scientific rigor and avoiding overclaiming.

---

### [2026-10-23] - Revision 1.2: Rigor Audit, Comparator Reconciliation & CKD Quarantine

- **Change 1: Programmatic Regeneration of All Summary Tables**
  - *Description*: Implemented `scripts/generate_markdown_tables.py` to regenerate all summary tables directly from `results/*.parquet` without manual patching. Old values audited and replaced with exact fold-aggregate values.
  - *Rationale*: Complete data reproducibility and elimination of manual copy errors.

- **Change 2: Weight-Jitter Evaluation Reframing to Jaccard-Only Metric**
  - *Description*: Removed the exploratory Logistic Regression ROC-AUC column from the weight-jitter sensitivity table; restricted reported evaluation strictly to Top-$K$ Jaccard stability relative to the unjittered reference ranking (self-identity Jaccard $\equiv 1.0000$ at $\eta=0$).
  - *Rationale*: The initial jitter evaluation fitted Logistic Regression on normalized cohort matrices without fold-wise cross-validation (violating strict in-fold isolation for test metrics). Jaccard stability accurately quantifies feature ranking robustness across 100 noise draws per level without conflating downstream discrimination.

- **Change 3: Systematic Correction of Non-Inferiority Failure Patterns**
  - *Description*: Corrected the Diabetes failure pattern characterization: all 10 failures ($10/10$, 100%) use the Random Forest classifier; 9 of 10 ($9/10$, 90%) occur at the tightest feature budget $K=5$; and 7 of 10 ($7/10$, 70%) use Reciprocal Rank Fusion (RRF). Removed the speculative "RF overfit" rationale in favor of documented dimension loss under severe feature compression and non-linear rank compression. Separated the 13 Heart Disease failures into genuine negative deltas ($\Delta \le 0$, 2 cases under ANOVA+LogReg+RRF at $K=3$) and underpowered cases where empirical deltas are positive ($\Delta > 0$, 11 cases) but non-inferiority is unestablished due to wide confidence intervals from small sample size ($N=303$).
  - *Rationale*: Objective and mathematically faithful failure auditing.

- **Change 4: Comparator Metrics Recomputation, Jaccard to Baseline, Frontier, and Diabetes Prior-mRMR Reconciliation**
  - *Description*: Recomputed hallmark feature share ($w \ge 0.8$) and downstream ROC-AUC with 95% confidence intervals within the same cross-validation evaluation loop across all three cohorts. Computed each method's Top-$K$ Jaccard stability to the unweighted baseline selector. Formally reconciled the Diabetes Prior-mRMR AUC: the full multi-classifier multi-budget grid benchmark achieves $0.7505 \pm 0.0496$ (due to acute degradation at $K=5$ where AUC dips to $0.6864$), whereas the linear sub-regime (LogReg / XGBoost at $K \ge 15$) achieves $0.7850 - 0.8109$.
  - *Rationale*: Direct head-to-head Pareto frontier evaluation with statistical confidence bounds.

- **Change 5: Explicit Declaration of Real Main Experimental Grid**
  - *Description*: Explicitly stated the exact parameter grid below each cohort summary table: Selectors = {ANOVA, LASSO, Random Forest}; Classifiers = {Logistic Regression, Random Forest}; Fusion Operators = {Hadamard, RRF}; $\lambda \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$; $K \in \{5, 10, 15, 20\}$ ($K \in \{3, 5, 8, 10\}$ for Heart Disease). Prohibited extrapolating selector-agnostic claims beyond the three selectors evaluated in the primary grid; documented that mRMR and XGBoost were evaluated in benchmark comparators and controls.
  - *Rationale*: Scientific accuracy and avoiding overgeneralized claims.

- **Change 6: Empirical Ranking Verification for Feature Promotion Claims**
  - *Description*: Verified all qualitative claims regarding feature promotion/demotion against empirical fold-level and cohort rankings. Documented that in Heart Disease, `cp` is promoted across all three selectors (rank #6 $\to$ #5 in ANOVA, #4 $\to$ #3 in LASSO, #2 $\to$ #1 in RF), while `sex` is demoted (rank #3 $\to$ #5 in LASSO), supporting the $+0.0060$ AUC elevation.
  - *Rationale*: Grounding qualitative explanatory statements in verifiable quantitative rank shifts.

- **Change 7: Quarantine of BD-KDD Chronic Kidney Disease (CKD) Artifacts and Provenance Audit**
  - *Description*: Conducted a forensic provenance investigation into `ckd_raw.csv` ($N=988$). Identified its origin as the **BD-KDD** dataset from Popular Diagnostic Center (Savar, Dhaka, Bangladesh; Zenodo), misattributed in exploratory literature as a 988-row variant of the canonical 400-patient Apollo Hospitals / UCI CKD benchmark. Performed clinical biomarker sanity testing: BD-KDD exhibits near-zero difference between Class 0 and Class 1 for Serum Creatinine ($7.41$ vs $7.65$ mg/dL) and Urine Albumin ($2.04$ vs $2.03$). Quarantined all CKD raw data, clean parquets, comparators, negative controls, jitter, tables, and figures into `data/quarantine/`. Added download URLs and SHA-256 hashes to all dataset cards, and implemented `tests/test_dataset_sanity.py` with passing unit tests.
  - *Rationale*: Preserving research integrity and data hygiene by isolating benchmark artifacts that lack clinical ground-truth signal.

---

### [2026-10-25] - Revision 1.4: Full 5-Selector, 3-Classifier, 4-Dataset Primary Grid Execution

- **Change 1: Full Implementation & Integration of All Five Feature Selectors**
  - *Description*: Implemented `BorutaSelector(BaseSelector)` with shadow-feature permutation and Random Forest feature importance scoring inside `doda/selectors/selectors.py`. Standardized `MRMRSelector(BaseSelector)` with mutual information relevance and pairwise correlation redundancy penalty. Both selectors are fully registered in `make_selector()` and exported via `doda/selectors/__init__.py`.
  - *Verification*: Added unit and leakage-safety test suite `tests/test_selectors.py` confirming feature name preservation, deterministic seeding, non-negative scores, and strict in-fold isolation. All 30 test cases pass.

- **Change 2: Execution of Complete Primary Benchmark Grid Across All Four Eligible Datasets**
  - *Description*: Executed the standardized primary benchmark across all 4 eligible cohorts (UCI Cleveland Heart Disease, WDBC Breast Cancer, Canonical UCI CKD, and CDC BRFSS Diabetes).
  - *Grid Structure*: 5 Selectors (ANOVA, LASSO, Random Forest, mRMR, Boruta) $\times$ 3 Classifiers (Logistic Regression, Random Forest, XGBoost) $\times$ 4 Top-$K$ Budgets $\times$ 5 Lambdas ($\lambda \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$) $\times$ 2 Fusion Operators (Hadamard, RRF) across 50 folds (5 splits $\times$ 10 repeats Repeated Stratified Cross-Validation).
  - *Evaluation Volume*: Exactly 30,000 evaluations per dataset ($120,000$ total evaluations across the 4 cohorts), saved cleanly into `results/grid_5sel_3clf/<dataset>/runs.parquet`. Historical baseline runs are preserved untouched.
  - *Checkpointing & Resume*: Added fold-checkpointing and `--start-repeat`/`--end-repeat` CLI arguments to `scripts/run_experiments.py` ensuring fault-tolerant execution and clean pause/resume workflows.

