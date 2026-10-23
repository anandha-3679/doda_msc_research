"""
TOST (Two One-Sided Tests) Statistical Equivalence Module.
Includes Nadeau-Bengio resampled cross-validation variance correction.
"""

from typing import Dict, Tuple
import numpy as np
import scipy.stats as stats


def nadeau_bengio_corrected_var(diffs: np.ndarray, n_train: int, n_test: int) -> float:
    """
    Nadeau-Bengio variance correction for repeated cross-validation:
    sigma^2_corrected = (1/J + n_test / n_train) * sample_variance
    where J = total number of resampled folds (e.g. 5 folds x 10 repeats = 50).
    """
    J = len(diffs)
    if J <= 1:
        return 0.0
    s2 = np.var(diffs, ddof=1)
    correction_factor = (1.0 / J) + (n_test / n_train)
    corrected_var = correction_factor * s2
    return corrected_var


def run_tost(
    diffs: np.ndarray,
    epsilon: float = 0.02,
    alpha: float = 0.05,
    n_train: int = 8000,
    n_test: int = 2000,
) -> Dict[str, float]:
    """
    Perform Two One-Sided Tests (TOST) on paired differences: Delta = metric_DODA - metric_baseline.
    
    H0_lower: Delta <= -epsilon
    H0_upper: Delta >= +epsilon
    
    Rejection of both H0 at level alpha establishes equivalence within [-epsilon, +epsilon].
    """
    diffs = np.asarray(diffs, dtype=float)
    mean_diff = np.mean(diffs)
    corr_var = nadeau_bengio_corrected_var(diffs, n_train=n_train, n_test=n_test)
    se = np.sqrt(corr_var) if corr_var > 0 else 1e-9

    df = len(diffs) - 1
    # Test 1: Delta > -epsilon -> (mean_diff - (-epsilon)) / se
    t_lower = (mean_diff - (-epsilon)) / se
    p_lower = 1.0 - stats.t.cdf(t_lower, df=df)

    # Test 2: Delta < +epsilon -> (+epsilon - mean_diff) / se
    t_upper = (epsilon - mean_diff) / se
    p_upper = 1.0 - stats.t.cdf(t_upper, df=df)

    p_tost = max(p_lower, p_upper)
    is_equivalent = bool(p_tost < alpha)

    # 1 - 2*alpha CI (e.g. 90% two-sided CI, where lower bound corresponds to 95% one-sided CI)
    t_crit = stats.t.ppf(1.0 - alpha, df=df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    # Primary RQ1: One-sided non-inferiority test (lower CI >= -epsilon, p_lower < alpha)
    is_non_inferior = bool(p_lower < alpha and ci_lower >= -epsilon)
    exceeds_upper_bound = bool(ci_lower > +epsilon)

    # Descriptive verdict
    if is_equivalent:
        verdict = "equivalent"
    elif exceeds_upper_bound:
        verdict = "non-inferior-and-higher"
    elif is_non_inferior:
        verdict = "non-inferior"
    else:
        verdict = "inferior"

    return {
        "mean_diff": float(mean_diff),
        "se_corrected": float(se),
        "p_non_inferior": float(p_lower),
        "p_tost": float(p_tost),
        "is_non_inferior": is_non_inferior,
        "is_equivalent": is_equivalent,
        "verdict": verdict,
        "ci_lower_90": float(ci_lower),
        "ci_upper_90": float(ci_upper),
        "epsilon": float(epsilon),
    }
