"""
Figure generation script for DODA research paper.
Generates publication-ready figures F1-F5 from results/<dataset>/runs.parquet:
- F1: Controllability Curve (ROC-AUC vs Lambda with TOST Equivalence Bounds)
- F2: Clinical Alignment Trade-off (Spearman Rank Correlation vs Lambda)
- F3: Jaccard Top-K Overlap Stability across Lambda
- F4: Performance by Feature Subset Size Top-K (K in [5, 10, 15, 20])
- F5: TOST Forest Plot (90% Confidence Intervals of Paired Differences)
"""

import argparse
import sys
from pathlib import Path

# Ensure root workspace is prioritized
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Publication aesthetic settings
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_figures(results_path: str = "results/diabetes/runs.parquet", out_dir: str = "docs/figures"):
    res_file = Path(results_path)
    if not res_file.exists():
        raise FileNotFoundError(f"Results file not found: {res_file}")

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {res_file}...")
    df = pd.read_parquet(res_file)

    # -------------------------------------------------------------
    # Figure 1: Predictive Equivalence across Lambda Sweep (ROC-AUC)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.lineplot(
        data=df,
        x="lambda",
        y="roc_auc",
        hue="fusion_method",
        style="classifier",
        markers=True,
        dashes=False,
        errorbar=("ci", 95),
        palette=["#1f77b4", "#ff7f0e"],
        ax=ax,
    )

    baseline_mean = df[df["lambda"] == 0.0]["roc_auc"].mean()
    ax.axhline(baseline_mean, color="black", linestyle="--", alpha=0.7, label=f"Baseline mean ({baseline_mean:.4f})")
    ax.axhspan(baseline_mean - 0.02, baseline_mean + 0.02, color="gray", alpha=0.15, label="TOST Margin (±0.02)")

    ax.set_title("F1: Predictive Equivalence across Clinical Weight Strength (λ)", fontsize=13, weight="bold")
    ax.set_xlabel("Clinical Weight Strength (λ)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(baseline_mean - 0.04, baseline_mean + 0.04)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    f1_path = out_p / "F1_predictive_equivalence_lambda.png"
    plt.savefig(f1_path)
    plt.close()
    print(f"Generated: {f1_path}")

    # -------------------------------------------------------------
    # Figure 2: RQ2 Clinical Alignment (Spearman Rank Correlation)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.lineplot(
        data=df,
        x="lambda",
        y="spearman_clinical_corr",
        hue="fusion_method",
        style="selector",
        markers=True,
        dashes=False,
        errorbar=None,
        palette=["#2ca02c", "#d62728"],
        ax=ax,
    )
    ax.set_title("F2: Clinical Alignment (Spearman ρ with Clinical Order vs λ)", fontsize=13, weight="bold")
    ax.set_xlabel("Clinical Weight Strength (λ)")
    ax.set_ylabel("Spearman Rank Correlation (ρ)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    f2_path = out_p / "F2_clinical_alignment_spearman.png"
    plt.savefig(f2_path)
    plt.close()
    print(f"Generated: {f2_path}")

    # -------------------------------------------------------------
    # Figure 3: Feature Top-K Ranking Stability (Jaccard Overlap)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.lineplot(
        data=df,
        x="lambda",
        y="jaccard_to_baseline",
        hue="top_k",
        palette="viridis",
        markers=True,
        dashes=False,
        errorbar=None,
        ax=ax,
    )
    ax.set_title("F3: Top-K Jaccard Overlap to Baseline across λ", fontsize=13, weight="bold")
    ax.set_xlabel("Clinical Weight Strength (λ)")
    ax.set_ylabel("Jaccard Similarity to Baseline Top-K")
    ax.set_ylim(0.6, 1.05)
    ax.legend(title="Top-K", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    f3_path = out_p / "F3_jaccard_overlap_stability.png"
    plt.savefig(f3_path)
    plt.close()
    print(f"Generated: {f3_path}")

    # -------------------------------------------------------------
    # Figure 4: Performance Across Top-K Settings
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.boxplot(
        data=df,
        x="top_k",
        y="roc_auc",
        hue="fusion_method",
        palette="Blues",
        ax=ax,
        fliersize=1,
    )
    ax.set_title("F4: Performance Distribution by Feature Subset Size (Top-K)", fontsize=13, weight="bold")
    ax.set_xlabel("Top-K Features Selected")
    ax.set_ylabel("ROC-AUC")
    ax.legend(title="Fusion Method", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    f4_path = out_p / "F4_top_k_performance_distribution.png"
    plt.savefig(f4_path)
    plt.close()
    print(f"Generated: {f4_path}")

    # -------------------------------------------------------------
    # Figure 5: TOST Forest Plot (90% Confidence Interval of Differences)
    # -------------------------------------------------------------
    tost_file = Path("docs/tables/tost_equivalence_analysis.csv")
    if tost_file.exists():
        tost_df = pd.read_csv(tost_file)
        # Select representative slice: K=10, lambda=1.0 across selectors and classifiers
        sub = tost_df[(tost_df["top_k"] == 10) & (tost_df["lambda"] == 1.0)].copy()
        sub["label"] = sub["selector"] + " + " + sub["classifier"] + " (" + sub["fusion_method"] + ")"

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        y_pos = np.arange(len(sub))

        ax.errorbar(
            sub["mean_diff_roc_auc"],
            y_pos,
            xerr=[sub["mean_diff_roc_auc"] - sub["ci_lower_90"], sub["ci_upper_90"] - sub["mean_diff_roc_auc"]],
            fmt="o",
            color="#0b5394",
            ecolor="#3d85c6",
            elinewidth=2,
            capsize=4,
            label="90% CI (Nadeau-Bengio corrected)",
        )

        ax.axvline(0.0, color="black", linestyle="-", linewidth=0.8)
        ax.axvline(-0.02, color="red", linestyle="--", linewidth=1.2, label="TOST Equivalence Bounds (±0.02)")
        ax.axvline(0.02, color="red", linestyle="--", linewidth=1.2)
        ax.axvspan(-0.02, 0.02, color="green", alpha=0.08, label="Equivalence Zone")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["label"])
        ax.set_xlabel("Paired Difference Δ (DODA - Baseline ROC-AUC)")
        ax.set_title("F5: TOST Equivalence Forest Plot (λ = 1.0, Top-10 Features)", fontsize=13, weight="bold")
        ax.set_xlim(-0.03, 0.03)
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        f5_path = out_p / "F5_tost_forest_plot.png"
        plt.savefig(f5_path)
        plt.close()
        print(f"Generated: {f5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication figures for DODA")
    parser.add_argument("--results", default="results/diabetes/runs.parquet", help="Path to runs.parquet")
    parser.add_argument("--out", default="docs/figures", help="Output directory for figures")

    args = parser.parse_args()
    generate_figures(args.results, args.out)
