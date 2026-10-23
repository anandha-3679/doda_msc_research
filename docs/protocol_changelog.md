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
  - *Description*: Standardized positioning wording: DODA is "selector-agnostic" (not "model-agnostic"). Prohibited claims of "proven zero-cost" or "enhanced stability". Provable weight provenance clarified as "guideline-derived (WHO/ADA/AHA/ACC/WDBC Consensus)".
  - *Rationale*: Scientific rigor and avoiding overclaiming.
