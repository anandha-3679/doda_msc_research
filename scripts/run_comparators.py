"""
Script to run Comparators (RQ3):
- Penalty-Factor LASSO (penalty = 1/w) vs DODA + LASSO
- Prior-Weighted mRMR vs DODA + mRMR

Evaluated across Heart Disease, WDBC, and Diabetes.
Saves to results/comparators/<dataset>_comparators.parquet.
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

from doda.knowledge import load_and_validate_weights
from doda.selectors import make_selector
from doda.fusion import fuse
from scripts.run_experiments import get_classifier, compute_metrics


def run_comparators(dataset_cfg_path: str, out_dir: str = "results/comparators"):
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

    X_all = df[feature_names].copy()
    y_all = df[target_col].values

    true_weights, _ = load_and_validate_weights(cfg["weights_path"], feature_names=feature_names)
    weights_vec = np.array([true_weights[f] for f in feature_names])

    top_ks = cfg["top_k"]
    classifiers = ["logistic_regression", "random_forest", "xgboost"]
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    results = []

    print(f"Running Comparators on {cfg['name']} (50 folds)...")

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_all, y_all), 1):
        repeat_num = (fold_idx - 1) // 5 + 1
        split_num = (fold_idx - 1) % 5 + 1

        X_train, y_train = X_all.iloc[train_idx].copy(), y_all[train_idx]
        X_test, y_test = X_all.iloc[test_idx].copy(), y_all[test_idx]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_norm = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X_train)), columns=feature_names)
        X_test_norm = pd.DataFrame(scaler.transform(imputer.transform(X_test)), columns=feature_names)

        # -------------------------------------------------------------
        # 1. LASSO Track: DODA+LASSO vs Penalty-Factor LASSO
        # -------------------------------------------------------------
        # A) Standard LASSO -> DODA (lambda=1.0)
        s_lasso = make_selector("lasso", random_state=seed)
        s_lasso.fit(X_train_norm, y_train)
        doda_lasso_scores = fuse(s_lasso.raw_math_scores_, true_weights, method="hadamard", strength=1.0).scores

        # B) Penalty-Factor LASSO: rescale columns X_j * w_j before standard L1
        # Smaller w_j compresses feature scale -> effectively increases L1 penalty for that feature
        X_train_rescaled = X_train_norm.multiply(weights_vec, axis=1)
        pf_lasso_model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=seed, max_iter=500)
        pf_lasso_model.fit(X_train_rescaled, y_train)
        pf_lasso_scores = pd.Series(np.abs(pf_lasso_model.coef_[0]), index=feature_names)

        # -------------------------------------------------------------
        # 2. mRMR Track: DODA+mRMR vs Prior-Weighted mRMR
        # -------------------------------------------------------------
        # A) Standard mRMR -> DODA (lambda=1.0)
        s_mrmr = make_selector("mrmr", random_state=seed)
        s_mrmr.fit(X_train_norm, y_train)
        doda_mrmr_scores = fuse(s_mrmr.raw_math_scores_, true_weights, method="hadamard", strength=1.0).scores

        # B) Prior-Weighted mRMR: relevance_j * w_j - 0.5 * redundancy_j
        rel = pd.Series(mutual_info_classif(X_train_norm, y_train, random_state=seed), index=feature_names)
        red = X_train_norm.corr().abs().mean(axis=1)
        prior_mrmr_scores = (rel * pd.Series(true_weights)) - 0.5 * red
        prior_mrmr_scores = prior_mrmr_scores - prior_mrmr_scores.min() + 1e-5

        methods = {
            "doda_lasso": doda_lasso_scores,
            "penalty_factor_lasso": pf_lasso_scores,
            "doda_mrmr": doda_mrmr_scores,
            "prior_weighted_mrmr": prior_mrmr_scores,
        }

        for method_name, scores in methods.items():
            for k in top_ks:
                top_features = list(scores.sort_values(ascending=False).head(k).index)
                high_weight_share = sum(1 for f in top_features if true_weights[f] >= 0.8) / k

                X_tr_k = X_train_norm[top_features]
                X_te_k = X_test_norm[top_features]

                for clf_name in classifiers:
                    clf = get_classifier(clf_name, seed=seed)
                    clf.fit(X_tr_k, y_train)
                    y_pred = clf.predict(X_te_k)
                    y_prob = clf.predict_proba(X_te_k)[:, 1] if hasattr(clf, "predict_proba") else y_pred

                    m = compute_metrics(y_test, y_pred, y_prob)

                    record = {
                        "dataset": cfg["name"],
                        "method": method_name,
                        "repeat": repeat_num,
                        "fold": split_num,
                        "top_k": k,
                        "classifier": clf_name,
                        "high_weight_share": float(high_weight_share),
                        **m,
                    }
                    results.append(record)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    file_path = out_p / f"{cfg['name']}_comparators.parquet"
    res_df.to_parquet(file_path, index=False)
    print(f"Saved {len(res_df)} comparator evaluations to {file_path}")
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Benchmark Comparators")
    parser.add_argument("--dataset", required=True, help="Dataset config path")
    parser.add_argument("--out", default="results/comparators", help="Output directory")

    args = parser.parse_args()
    run_comparators(args.dataset, args.out)
