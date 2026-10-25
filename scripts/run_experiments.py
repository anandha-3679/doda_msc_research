"""
End-to-end Fold-Safe Cross-Validation Experiment Runner for DODA.
Executes repeated stratified cross-validation on tabular clinical datasets.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure root workspace is prioritized over global site-packages
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hashlib
from typing import Dict, List, Optional
import numpy as np


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
import pandas as pd
import yaml
from scipy.stats import spearmanr

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    accuracy_score,
)

from doda.knowledge import load_and_validate_weights
from doda.selectors import make_selector
from doda.fusion import fuse


def get_classifier(name: str, seed: int = 42):
    name_clean = name.lower()
    if name_clean in ("logreg", "logistic_regression"):
        return LogisticRegression(max_iter=500, random_state=seed)
    elif name_clean in ("rf", "random_forest"):
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    elif name_clean == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier '{name}'")


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def run_experiment(
    dataset_cfg_path: str,
    n_splits: int = 5,
    n_repeats: int = 10,  # Standard 5x10 Repeated Stratified K-Fold
    start_repeat: int = 1,
    end_repeat: Optional[int] = None,
    selectors: List[str] = None,
    fusion_methods: List[str] = None,
    lambdas: List[float] = None,
    classifiers: List[str] = None,
    out_dir: str = "results/diabetes",
    sample_size: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["data_path"]
    weights_path = cfg["weights_path"]
    target_col = cfg["target"]
    top_k_list = cfg.get("top_k", [5, 10, 15, 20])
    feature_names = cfg["features"]
    end_rep = end_repeat or n_repeats

    selectors = selectors or ["anova", "lasso", "random_forest", "mrmr", "boruta"]
    fusion_methods = fusion_methods or ["hadamard", "rrf"]
    lambdas = lambdas if lambdas is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    classifiers = classifiers or ["logreg", "random_forest", "xgboost"]

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Subsampling if specified
    sample_n = sample_size or cfg.get("sample_size")
    if sample_n and sample_n < len(df):
        print(f"Sampling {sample_n} rows for fold evaluation (stratified, seed={seed})...")
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=sample_n, stratify=df[target_col], random_state=seed)

    X = df[feature_names].copy()
    y = df[target_col].copy()

    # Load and validate clinical weights
    weights, _ = load_and_validate_weights(weights_path, feature_names=feature_names)
    weight_series = pd.Series(weights)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    results = []

    total_folds = n_splits * n_repeats
    print(f"Starting {total_folds} CV folds ({n_splits} splits x {n_repeats} repeats)...")

    # Enforce isolated directory and configuration manifest check
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_file = out_path / "config_manifest.json"

    current_config = {
        "dataset_name": cfg["name"],
        "dataset_cfg_path": str(dataset_cfg_path),
        "data_path": str(data_path),
        "data_sha256": compute_file_sha256(Path(data_path)),
        "weights_path": str(weights_path),
        "weights_sha256": compute_file_sha256(Path(weights_path)),
        "n_splits": int(n_splits),
        "n_repeats": int(n_repeats),
        "selectors": sorted(selectors),
        "fusion_methods": sorted(fusion_methods),
        "lambdas": [float(x) for x in lambdas],
        "classifiers": sorted(classifiers),
        "top_k": list(top_k_list),
        "seed": int(seed),
        "sample_size": sample_n,
    }

    existing_repeats = list(out_path.glob("repeat_*.parquet"))
    if manifest_file.exists():
        import json
        with open(manifest_file, "r", encoding="utf-8") as mf:
            saved_config = json.load(mf)
        if saved_config != current_config:
            raise ValueError(
                f"Configuration mismatch in output directory '{out_dir}'! Existing runs were executed with a different config:\n"
                f"Saved: {saved_config}\n"
                f"Current: {current_config}\n"
                f"Use a clean or distinct --out directory to avoid mixing results."
            )
    elif existing_repeats:
        # Checkpoints exist but no manifest: warn or abort to prevent silent corruption
        raise RuntimeError(
            f"Output directory '{out_dir}' contains existing repeat files ({len(existing_repeats)} found) "
            f"without a config_manifest.json. Aborting to prevent silent mixing of incompatible configurations. "
            f"Please specify an isolated output directory or clean the existing folder."
        )
    else:
        import json
        with open(manifest_file, "w", encoding="utf-8") as mf:
            json.dump(current_config, mf, indent=2)

    fold_idx = 0
    start_time = time.time()

    for train_idx, test_idx in rskf.split(X, y):
        repeat_num = fold_idx // n_splits + 1
        split_num = fold_idx % n_splits + 1
        fold_idx += 1

        if repeat_num < start_repeat or repeat_num > end_rep:
            continue

        checkpoint_file = out_path / f"repeat_{repeat_num}.parquet"
        if checkpoint_file.exists():
            if fold_idx % n_splits == 0:
                print(f"Skipping repeat {repeat_num}/{n_repeats} (already completed and cached in {checkpoint_file.name})")
            continue

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # Step 2: In-Fold Transforms (Impute & Scale fit STRICTLY on train)
        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names, index=X_train.index)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_names, index=X_test.index)

        scaler = StandardScaler()
        X_train_norm = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_names, index=X_train.index)
        X_test_norm = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_names, index=X_test.index)

        # Iterate over selectors
        for sel_name in selectors:
            if sel_name == "anova":
                selector = make_selector(sel_name)
            else:
                selector = make_selector(sel_name, random_state=seed)
            selector.fit(X_train_norm, y_train)
            raw_scores = selector.raw_math_scores_

            # Base ranking (lambda=0 or baseline mathematical support)
            base_fused = fuse(raw_scores, weights, method="hadamard", strength=0.0)

            for fusion_method in fusion_methods:
                for lam in lambdas:
                    # Score fusion
                    fused = fuse(raw_scores, weights, method=fusion_method, strength=lam)

                    # Correlation with clinical order
                    spearman_corr, _ = spearmanr(fused.scores, weight_series)

                    for k in top_k_list:
                        top_features = fused.top_k(k)
                        base_top_features = base_fused.top_k(k)

                        # Jaccard overlap against baseline Top-K
                        jaccard = len(set(top_features) & set(base_top_features)) / len(
                            set(top_features) | set(base_top_features)
                        )

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
                                "repeat": repeat_num,
                                "fold": split_num,
                                "selector": sel_name,
                                "fusion_method": fusion_method,
                                "lambda": lam,
                                "top_k": k,
                                "classifier": clf_name,
                                "jaccard_to_baseline": jaccard,
                                "spearman_clinical_corr": float(spearman_corr),
                                **m,
                            }
                            results.append(record)

        if fold_idx % n_splits == 0:
            elapsed = time.time() - start_time
            print(f"Completed repeat {repeat_num}/{n_repeats} ({elapsed:.1f}s elapsed)")
            # Save checkpoint per completed repeat
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            checkpoint_file = out_path / f"repeat_{repeat_num}.parquet"
            pd.DataFrame(results).to_parquet(checkpoint_file, index=False)
            results = []

    # Final consolidation of all repeat checkpoints
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    chk_files = sorted(out_path.glob("repeat_*.parquet"))
    if chk_files:
        results_df = pd.concat([pd.read_parquet(f) for f in chk_files], ignore_index=True)
    elif results:
        results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame()

    parquet_file = out_path / "runs.parquet"
    csv_file = out_path / "runs.csv"
    if not results_df.empty:
        results_df.to_parquet(parquet_file, index=False)
        results_df.to_csv(csv_file, index=False)
    print(f"Saved {len(results_df)} run evaluations to {parquet_file} and {csv_file}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DODA Cross-Validation Experiments")
    parser.add_argument("--dataset", default="configs/datasets/diabetes.yaml", help="Path to dataset config")
    parser.add_argument("--splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats (standard 10 for 5x10 CV)")
    parser.add_argument("--start-repeat", type=int, default=1, help="First repeat to run (1-indexed)")
    parser.add_argument("--end-repeat", type=int, default=None, help="Last repeat to run (inclusive)")
    parser.add_argument("--out", default="results/diabetes", help="Output directory")
    parser.add_argument("--sample", type=int, default=None, help="Sample size override")

    args = parser.parse_args()
    run_experiment(
        dataset_cfg_path=args.dataset,
        n_splits=args.splits,
        n_repeats=args.repeats,
        start_repeat=args.start_repeat,
        end_repeat=args.end_repeat,
        out_dir=args.out,
        sample_size=args.sample,
    )
