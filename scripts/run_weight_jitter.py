"""
Script to run Weight-Jitter Sensitivity Analysis:
Multiplicative perturbation of weights: w' = clip(w * (1 + delta), 0, 1), delta in [-eta, +eta]
for eta in {0.10, 0.20}, 100 draws per setting with fixed reproducible seeds.

Outputs:
1. Top-K Jaccard stability vs unjittered ranking across all selectors (ANOVA, LASSO, RF, mRMR).
2. Performance non-inferiority check across draws using fast Logistic Regression.
Boruta dropped due to quadratic runtime.

Saves to results/sensitivity/<dataset>_jitter.parquet.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from doda.knowledge import load_and_validate_weights
from doda.selectors import make_selector
from doda.fusion import fuse


def run_weight_jitter(dataset_cfg_path: str, n_draws: int = 100, out_dir: str = "results/sensitivity"):
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(cfg["data_path"])
    target_col = cfg["target"]
    feature_names = [c for c in df.columns if c != target_col]
    seed = cfg.get("seed", 42)

    sample_size = cfg.get("sample_size", None)
    if sample_size and len(df) > sample_size:
        df = df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(int(np.rint(sample_size * len(x) / len(df))), random_state=seed)
        ).reset_index(drop=True)

    X = df[feature_names].copy()
    y = df[target_col].values

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X)), columns=feature_names)

    true_weights, _ = load_and_validate_weights(cfg["weights_path"], feature_names=feature_names)

    selectors = ["anova", "lasso", "random_forest", "mrmr"]
    top_ks = cfg["top_k"]
    noise_levels = [0.10, 0.20]

    # Pre-fit selectors once on full cohort to obtain base raw mathematical scores
    fitted_selectors = {}
    unjittered_top_k = {}
    for sel_name in selectors:
        sel = make_selector(sel_name, random_state=seed)
        sel.fit(X_norm, y)
        raw_scores = sel.raw_math_scores_
        fitted_selectors[sel_name] = raw_scores

        # Unjittered DODA Top-K (lambda = 1.0)
        base_fused = fuse(raw_scores, true_weights, method="hadamard", strength=1.0).scores
        for k in top_ks:
            unjittered_top_k[(sel_name, k)] = set(base_fused.sort_values(ascending=False).head(k).index)

    results = []
    print(f"Running Weight-Jitter Sensitivity on {cfg['name']} ({n_draws} draws x 2 noise levels)...")

    rng = np.random.default_rng(seed)

    for eta in noise_levels:
        for draw in range(1, n_draws + 1):
            # Draw multiplicative noise delta in [-eta, +eta]
            jittered_weights = {}
            for feat, w in true_weights.items():
                delta = rng.uniform(-eta, eta)
                jittered_weights[feat] = float(np.clip(w * (1.0 + delta), 0.0, 1.0))

            for sel_name, raw_scores in fitted_selectors.items():
                fused = fuse(raw_scores, jittered_weights, method="hadamard", strength=1.0).scores

                for k in top_ks:
                    top_feats = list(fused.sort_values(ascending=False).head(k).index)
                    base_feats = unjittered_top_k[(sel_name, k)]

                    # Jaccard stability vs unjittered ranking
                    jaccard = len(set(top_feats) & base_feats) / len(set(top_feats) | base_feats)

                    # Quick evaluation with Logistic Regression
                    clf = LogisticRegression(max_iter=500, random_state=seed)
                    clf.fit(X_norm[top_feats], y)
                    prob = clf.predict_proba(X_norm[top_feats])[:, 1]
                    auc = roc_auc_score(y, prob)

                    results.append({
                        "dataset": cfg["name"],
                        "eta": eta,
                        "draw": draw,
                        "selector": sel_name,
                        "top_k": k,
                        "jaccard_to_unjittered": jaccard,
                        "roc_auc_lr": auc,
                    })

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    file_path = out_p / f"{cfg['name']}_jitter.parquet"
    res_df.to_parquet(file_path, index=False)
    print(f"Saved {len(res_df)} jitter evaluations to {file_path}")
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Weight-Jitter Sensitivity Analysis")
    parser.add_argument("--dataset", required=True, help="Dataset config path")
    parser.add_argument("--draws", type=int, default=100, help="Number of random draws per noise level")
    parser.add_argument("--out", default="results/sensitivity", help="Output directory")

    args = parser.parse_args()
    run_weight_jitter(args.dataset, args.draws, args.out)
