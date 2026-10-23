"""
Experiments for Negative Controls (RQ3) across all three datasets:
- Shuffled / Permuted WHO weights (main control preserving weight multiset)
- Uniform Random weights
- Neutral weights (all 1.0)
- Inverted weights (w -> 1 - w, extreme stress test)

Protocol rules:
- Evaluated at lambda = 1.0, primary fusion (Hadamard), main K values
- All 5 selectors (ANOVA, LASSO, RF, mRMR, Boruta) and all 3 classifiers (LR, RF, XGBoost)
- Appended to results/negative_controls/runs.parquet (does not overwrite main grid)
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
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from doda.knowledge import load_and_validate_weights
from doda.selectors import make_selector
from doda.fusion import fuse
from scripts.run_experiments import get_classifier, compute_metrics


def run_negative_controls(dataset_cfg_path: str, out_dir: str = "results/negative_controls"):
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = Path(cfg["data_path"])
    df = pd.read_parquet(data_path)
    target_col = cfg["target"]
    feature_names = [c for c in df.columns if c != target_col]

    sample_size = cfg.get("sample_size", None)
    seed = cfg.get("seed", 42)

    if sample_size and len(df) > sample_size:
        df = df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(int(np.rint(sample_size * len(x) / len(df))), random_state=seed)
        ).reset_index(drop=True)

    X_all = df[feature_names].copy()
    y_all = df[target_col].values

    true_weights, _ = load_and_validate_weights(cfg["weights_path"], feature_names=feature_names)
    weight_vals = [true_weights[f] for f in feature_names]

    # Generate the 4 control profiles
    rng = np.random.default_rng(seed)
    
    # 1. Permuted weights: exact multiset preserved
    permuted_vals = rng.permutation(weight_vals).tolist()
    permuted_weights = dict(zip(feature_names, permuted_vals))

    # 2. Random weights: uniform in [0, 1]
    random_vals = rng.uniform(0.0, 1.0, size=len(feature_names)).tolist()
    random_weights = dict(zip(feature_names, random_vals))

    # 3. Neutral weights: all 1.0
    neutral_weights = {f: 1.0 for f in feature_names}

    # 4. Inverted weights: 1.0 - w
    inverted_weights = {f: float(np.clip(1.0 - true_weights[f], 0.0, 1.0)) for f in feature_names}

    controls = {
        "guideline_true": true_weights,
        "permuted_weights": permuted_weights,
        "random_weights": random_weights,
        "neutral_weights": neutral_weights,
        "inverted_weights": inverted_weights,
    }

    selectors = ["anova", "lasso", "random_forest", "mrmr"]
    classifiers = ["logistic_regression", "random_forest", "xgboost"]
    top_ks = cfg["top_k"]

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    results = []

    print(f"Running Negative Controls on {cfg['name']} (50 folds)...")
    start_time = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_all, y_all), 1):
        repeat_num = (fold_idx - 1) // 5 + 1
        split_num = (fold_idx - 1) % 5 + 1

        X_train, y_train = X_all.iloc[train_idx].copy(), y_all[train_idx]
        X_test, y_test = X_all.iloc[test_idx].copy(), y_all[test_idx]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        X_train_norm = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_names)
        X_test_norm = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_names)

        for sel_name in selectors:
            selector = make_selector(sel_name, random_state=seed)
            selector.fit(X_train_norm, y_train)
            raw_scores = selector.raw_math_scores_

            for ctrl_name, w_dict in controls.items():
                fused_result = fuse(
                    raw_scores=raw_scores,
                    weights=w_dict,
                    method="hadamard",
                    strength=1.0,
                )
                fused_scores = fused_result.scores

                # Spearman correlation against real clinical guideline profile
                spearman_corr, _ = spearmanr(
                    fused_scores.values,
                    [true_weights[f] for f in fused_scores.index]
                )

                for k in top_ks:
                    top_features = list(fused_scores.sort_values(ascending=False).head(k).index)
                    baseline_top = list(raw_scores.sort_values(ascending=False).head(k).index)

                    jaccard = len(set(top_features) & set(baseline_top)) / len(set(top_features) | set(baseline_top))

                    # Share of high-weight features (weight >= 0.8)
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
                            "control_type": ctrl_name,
                            "repeat": repeat_num,
                            "fold": split_num,
                            "selector": sel_name,
                            "fusion_method": "hadamard",
                            "lambda": 1.0,
                            "top_k": k,
                            "classifier": clf_name,
                            "jaccard_to_baseline": jaccard,
                            "spearman_guideline_corr": float(spearman_corr),
                            "high_weight_share": float(high_weight_share),
                            **m,
                        }
                        results.append(record)

        if fold_idx % 5 == 0:
            print(f"Completed repeat {repeat_num}/10 ({time.time() - start_time:.1f}s elapsed)")

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    
    file_path = out_p / f"{cfg['name']}_controls.parquet"
    res_df.to_parquet(file_path, index=False)
    print(f"Saved {len(res_df)} negative control evaluations to {file_path}")
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Negative Control Experiments")
    parser.add_argument("--dataset", required=True, help="Path to dataset yaml config")
    parser.add_argument("--out", default="results/negative_controls", help="Output directory")

    args = parser.parse_args()
    run_negative_controls(args.dataset, args.out)
