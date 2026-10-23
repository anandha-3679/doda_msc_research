"""
Experiments for WDBC Extreme Sparsity Track (K in {2, 3}):
Evaluates baseline (lambda=0.0) and DODA (lambda=1.0) under identical 5x10 CV folds.
Saves to results/breast_cancer_small_k/runs.parquet.
Reuses main grid for K=5.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from doda.knowledge import load_and_validate_weights
from doda.selectors import make_selector
from doda.fusion import fuse
from scripts.run_experiments import get_classifier, compute_metrics


def run_wdbc_small_k(out_dir: str = "results/breast_cancer_small_k"):
    cfg_path = "configs/datasets/breast_cancer.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(cfg["data_path"])
    target_col = cfg["target"]
    feature_names = [c for c in df.columns if c != target_col]
    seed = cfg.get("seed", 42)

    X_all = df[feature_names].copy()
    y_all = df[target_col].values

    true_weights, _ = load_and_validate_weights(cfg["weights_path"], feature_names=feature_names)

    small_ks = [2, 3]
    lambdas = [0.0, 1.0]
    selectors = ["anova", "lasso", "random_forest", "mrmr"]
    classifiers = ["logistic_regression", "random_forest", "xgboost"]

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    results = []

    print("Running WDBC Small-K Track (K in {2, 3}, 50 folds)...")

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_all, y_all), 1):
        repeat_num = (fold_idx - 1) // 5 + 1
        split_num = (fold_idx - 1) % 5 + 1

        X_train, y_train = X_all.iloc[train_idx].copy(), y_all[train_idx]
        X_test, y_test = X_all.iloc[test_idx].copy(), y_all[test_idx]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_norm = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X_train)), columns=feature_names)
        X_test_norm = pd.DataFrame(scaler.transform(imputer.transform(X_test)), columns=feature_names)

        for sel_name in selectors:
            selector = make_selector(sel_name, random_state=seed)
            selector.fit(X_train_norm, y_train)
            raw_scores = selector.raw_math_scores_

            for lam in lambdas:
                fused_result = fuse(
                    raw_scores=raw_scores,
                    weights=true_weights,
                    method="hadamard",
                    strength=lam,
                )
                fused_scores = fused_result.scores

                spearman_corr, _ = spearmanr(
                    fused_scores.values,
                    [true_weights[f] for f in fused_scores.index]
                )

                for k in small_ks:
                    top_features = list(fused_scores.sort_values(ascending=False).head(k).index)
                    baseline_top = list(raw_scores.sort_values(ascending=False).head(k).index)

                    jaccard = len(set(top_features) & set(baseline_top)) / len(set(top_features) | set(baseline_top))

                    X_tr_k = X_train_norm[top_features]
                    X_te_k = X_test_norm[top_features]

                    for clf_name in classifiers:
                        clf = get_classifier(clf_name, seed=seed)
                        clf.fit(X_tr_k, y_train)
                        y_pred = clf.predict(X_te_k)
                        y_prob = clf.predict_proba(X_te_k)[:, 1] if hasattr(clf, "predict_proba") else y_pred

                        m = compute_metrics(y_test, y_pred, y_prob)

                        record = {
                            "dataset": "breast_cancer",
                            "repeat": repeat_num,
                            "fold": split_num,
                            "selector": sel_name,
                            "fusion_method": "hadamard",
                            "lambda": lam,
                            "top_k": k,
                            "classifier": clf_name,
                            "jaccard_to_baseline": jaccard,
                            "spearman_clinical_corr": float(spearman_corr),
                            **m,
                        }
                        results.append(record)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    
    file_path = out_p / "runs.parquet"
    res_df.to_parquet(file_path, index=False)
    print(f"Saved {len(res_df)} small-K run evaluations to {file_path}")
    return res_df


if __name__ == "__main__":
    run_wdbc_small_k()
