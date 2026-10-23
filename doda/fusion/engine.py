"""
DODA Score Fusion Engine.
Implements:
1. Hadamard continuous multiplicative fusion: S_final(j) = S_norm(j) * w(j)^lambda
2. Reciprocal Rank Fusion (RRF): RRF(j) = 1/(k + r_math(j)) + lambda/(k + r_clin(j))
"""

from typing import Dict, List, Literal, Union
import numpy as np
import pandas as pd
from .normalization import normalize_scores


class FusedResult:
    def __init__(self, scores: pd.Series, ranks: pd.Series):
        self.scores = scores
        self.ranks = ranks

    def top_k(self, k: int) -> List[str]:
        """Return Top-K feature names ordered by descending fused score."""
        return list(self.scores.sort_values(ascending=False).head(k).index)


def fuse(
    raw_scores: Union[np.ndarray, pd.Series, dict],
    weights: Union[Dict[str, float], pd.Series],
    method: Literal["hadamard", "rrf"] = "hadamard",
    strength: float = 1.0,
    feature_names: List[str] = None,
    rrf_k: int = 60,
) -> FusedResult:
    """
    Fuse mathematical feature selection scores with clinical domain weights.
    
    Parameters:
    - raw_scores: Output scores from baseline feature selector.
    - weights: Clinical domain weights dictionary or Series.
    - method: 'hadamard' or 'rrf'.
    - strength: Lambda parameter in [0.0, 1.0].
    - feature_names: List of feature names if raw_scores is array.
    - rrf_k: Smoothing constant for RRF (default 60).
    """
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"Knowledge strength lambda must be in [0.0, 1.0], got {strength}")

    norm_s = normalize_scores(raw_scores, feature_names=feature_names)
    weights_s = pd.Series(weights).reindex(norm_s.index).fillna(1.0)

    if method == "hadamard":
        # S_final(j) = S_norm(j) * w(j)^lambda
        fused_s = norm_s * (weights_s ** strength)
    elif method == "rrf":
        # Rank features descending (1 = best)
        # Ties broken deterministically by index name
        r_math = norm_s.rank(ascending=False, method="first")
        r_clin = weights_s.rank(ascending=False, method="first")
        fused_s = (1.0 / (rrf_k + r_math)) + (strength / (rrf_k + r_clin))
    else:
        raise ValueError(f"Unsupported fusion method '{method}'. Choose 'hadamard' or 'rrf'.")

    # Ranks descending
    ranks = fused_s.rank(ascending=False, method="first")
    return FusedResult(scores=fused_s, ranks=ranks)
