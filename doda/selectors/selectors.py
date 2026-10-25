"""
Standardized feature selector interface and implementations.
Implements:
- anova (f_classif)
- lasso (LogisticRegression L1 or Lasso)
- random_forest (RandomForestClassifier feature importances)
- mrmr (Fast mutual information mRMR implementation)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class BaseSelector(ABC):
    def __init__(self, top_k: Optional[int] = None):
        self.top_k = top_k
        self.raw_math_scores_: Optional[pd.Series] = None
        self.feature_names_: Optional[List[str]] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseSelector":
        """Compute mathematical scores for all features in X."""
        pass

    def get_support(self, k: Optional[int] = None) -> List[str]:
        """Return Top-K feature names based on raw mathematical scores."""
        if self.raw_math_scores_ is None:
            raise ValueError("Selector must be fitted before calling get_support()")
        k_val = k or self.top_k or len(self.raw_math_scores_)
        return list(self.raw_math_scores_.sort_values(ascending=False).head(k_val).index)

    def transform(self, X: pd.DataFrame, k: Optional[int] = None) -> pd.DataFrame:
        """Select Top-K columns from X."""
        selected_cols = self.get_support(k=k)
        return X[selected_cols]


class ANOVASelector(BaseSelector):
    """ANOVA F-value feature selector."""
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ANOVASelector":
        self.feature_names_ = list(X.columns)
        f_scores, _ = f_classif(X, y)
        # Replace NaNs or infinite with 0.0
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        self.raw_math_scores_ = pd.Series(f_scores, index=self.feature_names_)
        return self


class LASSOSelector(BaseSelector):
    """L1-penalized Logistic Regression coefficient magnitude."""
    def __init__(self, top_k: Optional[int] = None, C: float = 1.0, random_state: int = 42):
        super().__init__(top_k=top_k)
        self.C = C
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LASSOSelector":
        self.feature_names_ = list(X.columns)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(
                penalty="l1",
                C=self.C,
                solver="liblinear",
                random_state=self.random_state,
                max_iter=500,
            )
            model.fit(X, y)
        coef_mag = np.abs(model.coef_[0])
        self.raw_math_scores_ = pd.Series(coef_mag, index=self.feature_names_)
        return self


class RandomForestSelector(BaseSelector):
    """Gini / Impurity-based feature importance from Random Forest."""
    def __init__(self, top_k: Optional[int] = None, n_estimators: int = 50, random_state: int = 42):
        super().__init__(top_k=top_k)
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestSelector":
        self.feature_names_ = list(X.columns)
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        self.raw_math_scores_ = pd.Series(rf.feature_importances_, index=self.feature_names_)
        return self


class MRMRSelector(BaseSelector):
    """
    Fast Minimum Redundancy Maximum Relevance (mRMR) selector.
    Computes mutual information relevance to y minus average mutual information redundancy.
    """
    def __init__(self, top_k: Optional[int] = None, random_state: int = 42):
        super().__init__(top_k=top_k)
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MRMRSelector":
        self.feature_names_ = list(X.columns)
        # Relevance: mutual information between each feature and target
        relevance = mutual_info_classif(X, y, random_state=self.random_state)
        rel_s = pd.Series(relevance, index=self.feature_names_)

        # Fast heuristic mRMR scoring:
        # Score_j = Relevance_j - Redundancy_penalty
        corr_matrix = X.corr().abs()
        redundancy = corr_matrix.mean(axis=1)

        mrmr_scores = rel_s - 0.5 * redundancy
        # Shift to positive domain
        mrmr_scores = mrmr_scores - mrmr_scores.min() + 1e-5
        self.raw_math_scores_ = mrmr_scores
        return self


class BorutaSelector(BaseSelector):
    """
    Boruta-inspired feature selector.
    Generates shadow (permuted) features, fits Random Forest, and measures the
    relative importance of actual features versus maximal shadow feature importance.
    Assigns continuous positive importance scores to self.raw_math_scores_.
    """
    def __init__(
        self,
        top_k: Optional[int] = None,
        n_estimators: int = 50,
        max_iter: int = 15,
        random_state: int = 42,
    ):
        super().__init__(top_k=top_k)
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BorutaSelector":
        self.feature_names_ = list(X.columns)
        n_features = len(self.feature_names_)
        hits = np.zeros(n_features, dtype=float)
        total_importance = np.zeros(n_features, dtype=float)
        rng = np.random.default_rng(self.random_state)

        for iteration in range(self.max_iter):
            # Create shadow features by shuffling each column
            X_shadow = X.copy()
            for col in X_shadow.columns:
                X_shadow[col] = rng.permutation(X_shadow[col].values)
            X_shadow.columns = [f"shadow_{c}" for c in self.feature_names_]

            # Combine real and shadow features
            X_combined = pd.concat([X, X_shadow], axis=1)

            # Fit Random Forest on combined dataset
            iter_seed = int(self.random_state + iteration * 100)
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                random_state=iter_seed,
                n_jobs=-1,
            )
            rf.fit(X_combined, y)

            importances = rf.feature_importances_
            real_imp = importances[:n_features]
            shadow_imp = importances[n_features:]

            max_shadow = np.max(shadow_imp) if len(shadow_imp) > 0 else 0.0
            hits += (real_imp > max_shadow).astype(float)
            total_importance += np.maximum(real_imp - max_shadow, 0.0) + real_imp

        # Final mathematical score: combination of hits and cumulative excess importance
        # Ensure scores are strictly non-negative and finite
        scores = hits + total_importance
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        # Shift slightly so min score is positive for normalization
        scores = scores - scores.min() + 1e-4

        self.raw_math_scores_ = pd.Series(scores, index=self.feature_names_)
        return self


def make_selector(name: str, top_k: Optional[int] = None, **kwargs) -> BaseSelector:
    name_clean = name.lower().replace("-", "_")
    if name_clean == "anova":
        return ANOVASelector(top_k=top_k)
    elif name_clean == "lasso":
        return LASSOSelector(top_k=top_k, **kwargs)
    elif name_clean in ("rf", "random_forest"):
        return RandomForestSelector(top_k=top_k, **kwargs)
    elif name_clean == "mrmr":
        return MRMRSelector(top_k=top_k, **kwargs)
    elif name_clean == "boruta":
        return BorutaSelector(top_k=top_k, **kwargs)
    else:
        raise ValueError(
            f"Unknown selector '{name}'. Supported: anova, lasso, random_forest, mrmr, boruta."
        )
