"""
Analysis and Reporting script for DODA evaluation results.
Computes TOST statistical equivalence, clinical correlation, and generates tables.
"""

import argparse
import sys
from pathlib import Path

# Ensure root workspace is prioritized over global site-packages
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from doda.stats import run_tost


def analyze_results(results_path: str = "results/diabetes/runs.parquet", out_dir: str = "docs/tables"):
    res_file = Path(results_path)
    if not res_file.exists():
        raise FileNotFoundError(f"Results file not found: {res_file}")

    df = pd.read_parquet(res_file)
    print(f"Loaded {len(df)} run evaluations.")

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Summary Performance Table by Selector, Classifier, Fusion, Lambda, Top-K
    summary = df.groupby(["selector", "fusion_method", "lambda", "top_k", "classifier"])[
        ["roc_auc", "f1", "balanced_accuracy", "pr_auc", "jaccard_to_baseline", "spearman_clinical_corr"]
    ].agg(["mean", "std"]).reset_index()

    summary_file = out_p / "performance_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved performance summary table to {summary_file}")

    # 2. TOST Equivalence Analysis: DODA (lambda > 0) vs Baseline (lambda == 0)
    tost_records = []
    selectors = df["selector"].unique()
    classifiers = df["classifier"].unique()
    fusion_methods = df["fusion_method"].unique()
    top_ks = df["top_k"].unique()
    lambdas = [l for l in df["lambda"].unique() if l > 0]

    for sel in selectors:
        for clf in classifiers:
            for fm in fusion_methods:
                for k in top_ks:
                    # Get baseline runs
                    base_mask = (
                        (df["selector"] == sel)
                        & (df["classifier"] == clf)
                        & (df["fusion_method"] == fm)
                        & (df["lambda"] == 0.0)
                        & (df["top_k"] == k)
                    )
                    base_df = df[base_mask].sort_values(["repeat", "fold"])

                    for lam in lambdas:
                        doda_mask = (
                            (df["selector"] == sel)
                            & (df["classifier"] == clf)
                            & (df["fusion_method"] == fm)
                            & (df["lambda"] == lam)
                            & (df["top_k"] == k)
                        )
                        doda_df = df[doda_mask].sort_values(["repeat", "fold"])

                        if len(base_df) > 0 and len(doda_df) == len(base_df):
                            diffs = doda_df["roc_auc"].values - base_df["roc_auc"].values
                            # Determine sample size for Nadeau-Bengio
                            dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns else ""
                            if "heart" in dataset_name.lower():
                                n_tr, n_te = 242, 61
                            elif "diabetes" in dataset_name.lower():
                                n_tr, n_te = 8000, 2000
                            elif "cancer" in dataset_name.lower() or "breast" in dataset_name.lower():
                                n_tr, n_te = 455, 114
                            elif "ckd" in dataset_name.lower():
                                n_tr, n_te = 320, 80
                            else:
                                n_tr, n_te = 800, 200

                            res_02 = run_tost(diffs, epsilon=0.02, n_train=n_tr, n_test=n_te)
                            res_01 = run_tost(diffs, epsilon=0.01, n_train=n_tr, n_test=n_te)

                            tost_records.append({
                                "dataset": dataset_name,
                                "selector": sel,
                                "classifier": clf,
                                "fusion_method": fm,
                                "lambda": lam,
                                "top_k": k,
                                "mean_diff_roc_auc": res_02["mean_diff"],
                                "se_corrected": res_02["se_corrected"],
                                "ci_lower_95_onesided": res_02["ci_lower_90"],
                                "ci_upper_90": res_02["ci_upper_90"],
                                "p_non_inferior_02": res_02["p_non_inferior"],
                                "is_non_inferior_02": res_02["is_non_inferior"],
                                "p_tost_02": res_02["p_tost"],
                                "is_equivalent_02": res_02["is_equivalent"],
                                "verdict_02": res_02["verdict"],
                                "p_non_inferior_01": res_01["p_non_inferior"],
                                "is_non_inferior_01": res_01["is_non_inferior"],
                                "p_tost_01": res_01["p_tost"],
                                "is_equivalent_01": res_01["is_equivalent"],
                            })

    tost_df = pd.DataFrame(tost_records)
    if len(tost_df) > 0:
        # Benjamini-Hochberg FDR adjustment
        from scipy.stats import false_discovery_control
        tost_df["p_non_inferior_02_fdr"] = false_discovery_control(tost_df["p_non_inferior_02"].values)
        tost_df["p_tost_02_fdr"] = false_discovery_control(tost_df["p_tost_02"].values)
        tost_df["is_non_inferior_02_fdr"] = tost_df["p_non_inferior_02_fdr"] < 0.05
        tost_df["is_equivalent_02_fdr"] = tost_df["p_tost_02_fdr"] < 0.05

    tost_file = out_p / "tost_equivalence_analysis.csv"
    tost_df.to_csv(tost_file, index=False)
    print(f"Saved non-inferiority & TOST results to {tost_file}")
    print(f"Total comparisons: {len(tost_df)}.")
    print(f"Primary Non-inferiority rate (eps=0.02, FDR): {tost_df['is_non_inferior_02_fdr'].mean():.2%}")
    print(f"Secondary TOST Equivalence rate (eps=0.02, FDR): {tost_df['is_equivalent_02_fdr'].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DODA Experiment Results")
    parser.add_argument("--results", default="results/diabetes/runs.parquet", help="Path to runs.parquet")
    parser.add_argument("--out", default="docs/tables", help="Output directory")

    args = parser.parse_args()
    analyze_results(args.results, args.out)
