# Generic Dataset Experiment Pipeline & Protocol

A standardized, leak-free experimental pipeline template for evaluating **DODA** across any tabular medical dataset. Every stage is identical across datasets except for the static preprocessing step (Step 1).

---

## Architecture Flow

```mermaid
flowchart TD
    Raw[Raw Dataset: data/raw/<dataset>_raw.csv] --> Preproc[1. Dataset-Specific Static Preprocessing]
    Preproc --> Clean[Clean Matrix: data/processed/<dataset>_clean.parquet]
    
    Clean --> CVLoop[2. Repeated Stratified K-Fold CV (5x10)]
    Weights[Frozen Weights: configs/weights/<dataset>_weights.json] --> CVLoop

    subgraph Fold Execution (Inside Training Fold Only)
        TrainX[Train Split X, y] --> InFoldPreproc[Imputation & Scaling fit on Train]
        InFoldPreproc --> Sel[Selectors: ANOVA / LASSO / mRMR / RF / Boruta]
        Sel --> RawScores[Raw Math Scores S_raw]
        RawScores --> Norm[Min-Max Normalization to [0,1]]
        Norm --> Fusion[Fusion Layer: Hadamard / RRF with λ in [0, 1]]
        Weights --> Fusion
        Fusion --> Ranked[Ranked Features & Top-K Subset Selection]
        Ranked --> Classifiers[Train Classifiers: LogReg / RF / XGBoost]
    end

    Classifiers --> TestEval[Evaluate on Test Split Fold]
    TestEval --> Results[3. Collect Results: results/<dataset>/runs.parquet]
    Results --> Stats[4. Statistical Analysis: TOST ±ε, BH FDR, Spearman ρ]
    Stats --> Reports[5. Tables & Figures Generation]
```

---

## Standardized Step-by-Step Pipeline

### Step 1: Dataset-Specific Preprocessing (Variable per dataset)
*File: `scripts/preprocess_<dataset>.py` or `doda/datasets/<dataset>.py`*
- Standardize column names into snake_case.
- Map and binarize target variable ($y \in \{0, 1\}$).
- Drop identifiers, leakage columns, and constant features.
- Output clean artifact: `data/processed/<dataset>_clean.parquet`.
- Document dataset metadata in `docs/data_cards/<dataset>.md`.

---

### Step 2: Weight Specification & Validation (Identical Interface)
*File: `configs/weights/<dataset>_weights.json`*
- Keys must strictly match columns in `data/processed/<dataset>_clean.parquet`.
- Values must satisfy $w_j \in [0, 1]$ (unlisted features default to 1.0).
- Schema validation via `doda/knowledge/validator.py`.
- Fill provenance in `configs/weights/provenance.csv` with primary guideline sources (e.g. ADA, AHA/ACC, KDIGO, WHO) before running experiments.

```json
{
  "_meta": {
    "dataset": "<dataset>",
    "version": "1.0",
    "source": "<Guideline Source>",
    "default_weight": 1.0
  },
  "weights": {
    "feature_name": 0.85
  }
}
```

---

### Step 3: Leakage-Free Fold-Safe Cross-Validation (Identical Engine)
*File: `doda/pipeline.py` & `scripts/run_experiments.py`*

For each repeat $r \in [1, 10]$ and fold $k \in [1, 5]$ (Repeated Stratified K-Fold):
1. **Partition**: Split $(X, y)$ into $(X_{\text{train}}, y_{\text{train}})$ and $(X_{\text{test}}, y_{\text{test}})$.
2. **In-Fold Transforms**:
   - Fit `SimpleImputer` on $X_{\text{train}}$, transform $X_{\text{train}}$ and $X_{\text{test}}$.
   - Fit `StandardScaler` / `MinMaxScaler` on $X_{\text{train}}$, transform $X_{\text{train}}$ and $X_{\text{test}}$.
3. **Feature Scoring**:
   - Fit selector on $X_{\text{train}}, y_{\text{train}}$ to obtain $S_{\text{raw}}$.
4. **Score Normalization**:
   - Map $S_{\text{raw}}$ to $S_{\text{norm}} \in [0, 1]$ using Min-Max.
5. **Score Fusion**:
   - **Hadamard**: $S_{\text{final}}(j) = S_{\text{norm}}(j) \cdot w(j)^\lambda$
   - **Reciprocal Rank Fusion (RRF)**: $RRF(j) = \frac{1}{60 + r_{\text{math}}(j)} + \frac{\lambda}{60 + r_{\text{clin}}(j)}$
   - Sweep $\lambda \in [0.0, 0.25, 0.50, 0.75, 1.0]$ to trace the Pareto trade-off frontier.
   - *Note on Default $\lambda^*$*: Any recommended operational default $\lambda^*$ must be selected via inner nested CV on $X_{\text{train}}$, never by inspecting test fold results.
6. **Top-$K$ Subset**:
   - Extract Top-$K$ columns from $X_{\text{train}}$ and apply the same mask to $X_{\text{test}}$.
7. **Downstream Classification**:
   - Train `LogisticRegression`, `RandomForestClassifier`, and `XGBClassifier` on $X_{\text{train}}[\text{Top-}K]$.
   - Compute predictions and probabilities on $X_{\text{test}}[\text{Top-}K]$.
8. **Logging**:
   - Record metrics (ROC-AUC, F1, Balanced Accuracy, PR-AUC).
   - Record ranking shifts (Jaccard similarity to baseline Top-$K$, Spearman $\rho$ against clinical weights).

---

### Step 4: Statistical Testing & Equivalence Analysis (Identical Engine)
*File: `doda/stats/equivalence.py` & `scripts/make_tables.py`*
- **Primary RQ1 (Equivalence & Non-inferiority)**:
  - Run **TOST** (Two One-Sided Tests) on paired differences: $\Delta = \text{metric}_{\text{DODA}} - \text{metric}_{\text{baseline}}$
  - Safety non-inferiority boundary: $\Delta \ge -\epsilon$ (demonstrates zero predictive harm from clinical weighting).
  - Bounds: $\epsilon = \pm 0.02$ (primary), $\epsilon = \pm 0.01$ (robustness check; noted as statistically underpowered on cohorts with $N \approx 300$).
  - Apply **Nadeau–Bengio resampled t-test correction** with fold-appropriate $n_{\text{train}}$ and $n_{\text{test}}$ to account for CV correlation.
- **RQ2 (Controllability & Trade-off Curve)**:
  - Verify monotonic Spearman rank correlation between final ranking and clinical priority profile across $\lambda \in [0, 1]$.
  - Characterize the descriptive Pareto frontier (Spearman $\rho$ vs. $\Delta \text{ROC-AUC}$).
- **RQ3 (Comparators & Controls)**:
  - Compare against Penalty-Factor LASSO, Prior-Weighted mRMR, and randomized/shuffled weight controls.
- **Multiple Testing Correction**:
  - Apply Benjamini–Hochberg False Discovery Rate (FDR) control across evaluated combinations.

---

### Step 5: Results & Reporting Generation (Identical Engine)
*Files: `scripts/make_tables.py` and `scripts/make_figures.py`*
- Automatically load `results/<dataset>/runs.parquet`.
- Generate tables (T1–T7) and figures (F1–F6) in `docs/tables/` and `docs/figures/`.

---

## Dataset Integration Checklist

When onboarding any new dataset:
- [ ] Create `scripts/preprocess_<dataset>.py` and generate clean data.
- [ ] Create `configs/datasets/<dataset>.yaml` defining target, features, and $K$ values.
- [ ] Create `configs/weights/<dataset>_weights.json` and document entries in `provenance.csv`.
- [ ] Create `docs/data_cards/<dataset>.md`.
- [ ] Run `python scripts/run_experiments.py --dataset configs/datasets/<dataset>.yaml`.
