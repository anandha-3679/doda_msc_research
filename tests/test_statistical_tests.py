"""
Tests for Non-Inferiority, TOST, Weight Transformations, and Jitter.
"""

import numpy as np
import pytest
from doda.stats.equivalence import run_tost, nadeau_bengio_corrected_var


def test_tost_known_equivalent():
    # Synthetic diffs identically zero
    diffs = np.zeros(50)
    res = run_tost(diffs, epsilon=0.02, alpha=0.05, n_train=800, n_test=200)
    assert res["is_non_inferior"] is True
    assert res["is_equivalent"] is True
    assert res["verdict"] == "equivalent"
    assert res["p_non_inferior"] < 0.05


def test_tost_known_inferior():
    # Synthetic diffs clearly below -epsilon (-0.05 vs epsilon 0.02)
    diffs = np.full(50, -0.05)
    res = run_tost(diffs, epsilon=0.02, alpha=0.05, n_train=800, n_test=200)
    assert res["is_non_inferior"] is False
    assert res["is_equivalent"] is False
    assert res["verdict"] == "inferior"


def test_tost_non_inferior_and_higher():
    # Synthetic diffs clearly above +epsilon (+0.05 vs epsilon 0.02)
    diffs = np.full(50, 0.05)
    res = run_tost(diffs, epsilon=0.02, alpha=0.05, n_train=800, n_test=200)
    assert res["is_non_inferior"] is True
    assert res["is_equivalent"] is False
    assert res["verdict"] == "non-inferior-and-higher"


def test_permuted_weights_preserves_multiset():
    weights = {"f1": 0.9, "f2": 0.5, "f3": 0.2, "f4": 0.9}
    keys = list(weights.keys())
    vals = list(weights.values())
    
    rng = np.random.default_rng(42)
    permuted_vals = rng.permutation(vals).tolist()
    perm_weights = dict(zip(keys, permuted_vals))

    # Weight multiset must be identical
    assert sorted(weights.values()) == sorted(perm_weights.values())


def test_inverted_weights_formula():
    weights = {"f1": 0.9, "f2": 0.5, "f3": 0.2, "f4": 1.0}
    inverted = {k: 1.0 - v for k, v in weights.items()}
    assert inverted["f1"] == pytest.approx(0.1)
    assert inverted["f2"] == pytest.approx(0.5)
    assert inverted["f3"] == pytest.approx(0.8)
    assert inverted["f4"] == pytest.approx(0.0)


def test_weight_jitter_bounds_and_reproducibility():
    weights = {"f1": 0.9, "f2": 0.1, "f3": 0.5}
    eta = 0.20
    
    rng1 = np.random.default_rng(123)
    jittered1 = {}
    for k, w in weights.items():
        delta = rng1.uniform(-eta, eta)
        jittered1[k] = float(np.clip(w * (1.0 + delta), 0.0, 1.0))

    rng2 = np.random.default_rng(123)
    jittered2 = {}
    for k, w in weights.items():
        delta = rng2.uniform(-eta, eta)
        jittered2[k] = float(np.clip(w * (1.0 + delta), 0.0, 1.0))

    # Reproducibility
    assert jittered1 == jittered2
    # Bounds check
    for v in jittered1.values():
        assert 0.0 <= v <= 1.0
