# Architecture Overview

DODA (Decoupled Operator-Driven Architecture) is a selector-agnostic framework designed to inject auditable, domain-informed weights between feature selector mathematical scores and final Top-K selection. It tests whether clinician-guided prioritization maintains predictive performance within a pre-declared statistical equivalence margin (±ε) using TOST.

### Key Pipeline Components
- **Selectors**: Interface for baseline mathematical feature selectors (ANOVA, LASSO, mRMR, Random Forest, Boruta).
- **Normalization**: Rescales heterogeneous mathematical scores to [0, 1].
- **Domain Weighting**: External, versioned JSON weight profiles with documented clinical evidence provenance.
- **Fusion**: Hadamard multiplication ($S \cdot w^\lambda$) or Reciprocal Rank Fusion (RRF).
- **Evaluation**: Leakage-free repeated stratified k-fold cross-validation, TOST equivalence analysis, and calibration checks against comparators/controls.

---

# Immediate Next Steps

<!-- Atomic, actionable checklist items ready for execution -->

- [x] Place uncleaned dataset in `data/raw/` and clinical weights in `configs/weights/`
- [x] Implement `scripts/preprocess_diabetes.py` to produce clean data matrix
- [x] Create dataset card `docs/data_cards/diabetes.md` and dataset configuration `configs/datasets/diabetes.yaml`
- [x] Verify weight schema against clean dataset features
- [x] Run end-to-end fold-safe evaluation loop according to `docs/pipeline_template.md`
- [x] Run full 5x10 repeated cross-validation grid (12,000 evaluations) on diabetes
- [x] Generate publication figures (F1-F5) and statistical equivalence tables for diabetes
- [x] Onboard UCI Heart Disease dataset (`processed.cleveland.data` and `heartdisease_weights.json`)
- [x] Run full 5x10 repeated cross-validation grid (12,000 evaluations) on heartdisease
- [x] Generate publication figures (F1-F5) and statistical equivalence tables for heartdisease
- [x] Resolve dataset size and weight provenance inconsistencies across README and docs
- [x] Update TOST analysis to record non-inferiority safety margin ($\Delta \ge -0.02$) and dataset-appropriate Nadeau-Bengio degrees of freedom
- [ ] Onboard CKD dataset (`data/raw/ckd_raw.csv` and `configs/weights/ckd_weights.json`)
- [x] Onboard Breast Cancer dataset (`data/raw/breast_cancer_raw` and `configs/weights/breast_cancer_weights.json`)
- [ ] Run cross-dataset synthesis report

---

# Completed Tasks

<!-- Completed checklist items -->
- [x] Initialized project repository configuration (`.antigravityrules`, `pyproject.toml`, `.gitignore`)
- [x] Created generic dataset execution template in `docs/pipeline_template.md`
- [x] Structured repository folders (`data/raw/`, `data/processed/`, `configs/weights/`, `configs/datasets/`, `doda/`, `scripts/`, `docs/`)
- [x] Transferred and standardized `diabetes.csv` into `data/raw/diabetes_raw.csv` and generated `data/processed/diabetes_clean.parquet`
- [x] Standardized `clinical_weights.json` into schema-compliant `configs/weights/diabetes_weights.json` with `configs/weights/provenance.csv`
- [x] Implemented core DODA modules: `doda.knowledge`, `doda.fusion`, `doda.selectors`, and `doda.stats`
- [x] Implemented and verified unit tests (`tests/test_weights.py`, `tests/test_fusion.py`)
- [x] Executed fold-safe cross-validation runner (`scripts/run_experiments.py`) and TOST statistical analysis (`scripts/make_tables.py`)
- [x] Standardized `processed.cleveland.data` -> `data/raw/heartdisease_raw.data` and generated `data/processed/heartdisease_clean.parquet`
- [x] Standardized `heartdisease_weights.json` -> `configs/weights/heartdisease_weights.json` and documented clinical rationale in `configs/weights/provenance.csv`
- [x] Created `configs/datasets/heartdisease.yaml` and `docs/data_cards/heartdisease.md`
- [x] Executed 5x10 repeated stratified CV grid (12,000 runs) saved to `results/heartdisease/runs.parquet`
- [x] Generated heartdisease statistical analysis tables (`docs/tables/heartdisease/`) and publication figures (`docs/figures/heartdisease/F1-F5`)
- [x] Reran diabetes statistical analysis with dataset-specific Nadeau-Bengio variance correction ($n_{\text{train}}=8000, n_{\text{test}}=2000$) and computed non-inferiority rates
- [x] Preprocessed WDBC dataset (`wdbc.data`) -> `data/processed/breast_cancer_clean.parquet` (569 samples, 30 features)
- [x] Created `configs/datasets/breast_cancer.yaml`, `docs/data_cards/breast_cancer.md`, and updated `configs/weights/provenance.csv` (64 total entries)
- [x] Executed full 5x10 cross-validation grid (12,000 runs) saved to `results/breast_cancer/runs.parquet`
- [x] Generated WDBC statistical equivalence tables (`docs/tables/breast_cancer/`) and publication figures (`docs/figures/breast_cancer/F1-F5`) (100% TOST equivalence and non-inferiority)
- [x] Executed Negative Controls (Permuted, Random, Neutral, Inverted) across all 3 datasets (36,000 evaluations) saved to `results/negative_controls/`
- [x] Executed Benchmark Comparators (Penalty-Factor LASSO vs DODA+LASSO, Prior-Weighted mRMR vs DODA+mRMR) across all 3 datasets (7,200 evaluations) saved to `results/comparators/`
- [x] Executed Weight-Jitter Sensitivity Analysis (100 draws x 2 noise levels) across all 3 datasets (9,600 evaluations) saved to `results/sensitivity/`
- [x] Executed WDBC Small-K Sparsity Track ($K \in \{2, 3\}$, 2,400 evaluations) saved to `results/breast_cancer_small_k/runs.parquet`
- [x] Formally documented protocol revisions (`docs/protocol.md`, `docs/protocol_changelog.md`, `docs/weight_provenance.md`) and verified unit tests (`tests/test_statistical_tests.py`)



