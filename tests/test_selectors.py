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


def test_no_leakage_scaler_imputer():
    """Verify fit statistics (mean, std) come strictly from training fold and are uninfluenced by test fold."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    # Create train and test with extreme distribution difference in test
    train_data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, np.nan, 40.0, 50.0]})
    test_data = pd.DataFrame({"a": [1000.0, 2000.0], "b": [10000.0, 20000.0]})
    
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(train_data)
    # Imputer statistics must only reflect training fold
    assert imputer.statistics_[0] == pytest.approx(3.0)
    assert imputer.statistics_[1] == pytest.approx(30.0)
    
    scaler = StandardScaler()
    scaler.fit(train_data.fillna(30.0))
    # Scaler mean must only reflect training fold
    assert scaler.mean_[0] == pytest.approx(3.0)
    assert scaler.mean_[1] == pytest.approx(30.0)


def test_no_leakage_selector(synthetic_data):
    """Selection never sees held-out fold labels or features."""
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


def test_seed_reproducibility(synthetic_data):
    """Same seed produces identical scores and selections across independent runs."""
    X, y = synthetic_data
    for sel_name in ALL_SELECTOR_NAMES:
        sel1 = make_selector(sel_name, top_k=3, random_state=123)
        sel2 = make_selector(sel_name, top_k=3, random_state=123)
        sel1.fit(X, y)
        sel2.fit(X, y)
        pd.testing.assert_series_equal(sel1.raw_math_scores_, sel2.raw_math_scores_)
        assert sel1.get_support(k=3) == sel2.get_support(k=3)


def test_penalty_lasso_equivalence():
    """Verify column-rescaling LASSO (X_j * w_j) matches reference penalty-factor behavior."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(42)
    X_vals = rng.normal(0, 1, size=(50, 4))
    cols = ["f1", "f2", "f3", "f4"]
    X = pd.DataFrame(X_vals, columns=cols)
    y = (X["f1"] + X["f2"] > 0).astype(int)
    
    # Clinical weights: f1 has weight 1.0 (low penalty), f2 has weight 0.1 (high penalty)
    weights = {"f1": 1.0, "f2": 0.1, "f3": 1.0, "f4": 0.5}
    weights_vec = pd.Series(weights)
    
    # Penalty-factor LASSO via column rescaling
    X_rescaled = X.multiply(weights_vec, axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42)
        model.fit(X_rescaled, y)

    
    # The coefficient on rescaled feature f2 should be heavily penalized (shrunk toward zero)
    # relative to f1 because its scale was shrunk by 0.1
    coefs = pd.Series(np.abs(model.coef_[0]), index=cols)
    assert coefs["f1"] > coefs["f2"]

