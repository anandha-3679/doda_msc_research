"""
DODA Score Normalization module.
Maps heterogeneous raw selector scores to [0, 1] range.
"""

from typing import Union
import numpy as np
import pandas as pd


def normalize_scores(
    scores: Union[np.ndarray, pd.Series, dict],
    feature_names: list = None,
) -> pd.Series:
    """
    Min-max normalizes scores to [0, 1].
    
    Edge case: If all scores are identical, returns a constant vector of 1.0.
    """
    if isinstance(scores, dict):
        s = pd.Series(scores)
    elif isinstance(scores, pd.Series):
        s = scores.copy()
    else:
        arr = np.asarray(scores, dtype=float)
        if feature_names is not None:
            s = pd.Series(arr, index=feature_names)
        else:
            s = pd.Series(arr)

    s_min = s.min()
    s_max = s.max()

    if np.isclose(s_max, s_min) or np.isnan(s_min) or np.isnan(s_max):
        # All identical or degenerate
        return pd.Series(1.0, index=s.index)

    norm_s = (s - s_min) / (s_max - s_min)
    return norm_s
