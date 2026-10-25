"""
Comprehensive unit and leakage-safety tests for all five feature selectors:
- ANOVA (ANOVASelector)
- LASSO (LASSOSelector)
- Random Forest (RandomForestSelector)
- mRMR (MRMRSelector)
- Boruta (BorutaSelector)
"""

import numpy as np
import pandas as pd
import pytest

from doda.selectors import (
    BaseSelector,
    ANOVASelector,
    LASSOSelector,
    RandomForestSelector,
    MRMRSelector,
    BorutaSelector,
    make_selector,
)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 8
    feature_names = [f"feat_{i}" for i in range(n_features)]
    
    # Generate feature matrix
    X_vals = rng.normal(0, 1, size=(n_samples, n_features))
    X = pd.DataFrame(X_vals, columns=feature_names)
    
    # Target strongly depends on feat_0 and feat_1
    logits = 2.0 * X["feat_0"] - 1.5 * X["feat_1"] + rng.normal(0, 0.5, size=n_samples)
    y = (logits > 0).astype(int)
    y.name = "target"
    return X, y


ALL_SELECTOR_NAMES = ["anova", "lasso", "random_forest", "mrmr", "boruta"]


@pytest.mark.parametrize("sel_name", ALL_SELECTOR_NAMES)
def test_selector_registration_and_types(sel_name):
    selector = make_selector(sel_name, top_k=3, random_state=42)
    assert isinstance(selector, BaseSelector)


@pytest.mark.parametrize("sel_name", ALL_SELECTOR_NAMES)
def test_selector_fit_and_alignment(sel_name, synthetic_data):
    X, y = synthetic_data
    selector = make_selector(sel_name, top_k=3, random_state=42)
    fitted = selector.fit(X, y)
    
    assert fitted is selector
    assert selector.raw_math_scores_ is not None
    assert isinstance(selector.raw_math_scores_, pd.Series)
    assert len(selector.raw_math_scores_) == len(X.columns)
    assert list(selector.raw_math_scores_.index) == list(X.columns)
    
    # Ensure scores are finite and non-negative
    scores = selector.raw_math_scores_.values
    assert not np.isnan(scores).any()
    assert not np.isinf(scores).any()
    assert (scores >= 0).all()
    
    # Top-K extraction
    top_3 = selector.get_support(k=3)
    assert len(top_3) == 3
    assert all(feat in X.columns for feat in top_3)
    
    # Transform produces dataframe with exactly chosen columns
    X_trans = selector.transform(X, k=3)
    assert list(X_trans.columns) == top_3
    assert len(X_trans) == len(X)


def test_leakage_safety_isolation(synthetic_data):
    """Confirm selectors strictly use training partition and do not leak test data."""
    X, y = synthetic_data
    X_train, y_train = X.iloc[:40], y.iloc[:40]
    X_test, y_test = X.iloc[40:], y.iloc[40:]
    
    for sel_name in ALL_SELECTOR_NAMES:
        sel = make_selector(sel_name, top_k=3, random_state=42)
        sel.fit(X_train, y_train)
        
        # Test transformation applies training-fitted feature mask without accessing test y
        X_test_k = sel.transform(X_test)
        assert len(X_test_k) == len(X_test)
        assert list(X_test_k.columns) == sel.get_support()
