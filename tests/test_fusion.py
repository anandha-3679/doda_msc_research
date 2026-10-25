"""
Unit tests for DODA score normalization and fusion logic.
Includes mandatory test verifying lambda=0 identity.
"""

import numpy as np
import pandas as pd
import pytest
from doda.fusion import normalize_scores, fuse


def test_normalization_bounds():
    scores = {"f1": 10.0, "f2": 5.0, "f3": 0.0}
    norm = normalize_scores(scores)
    assert norm["f1"] == 1.0
    assert norm["f2"] == 0.5
    assert norm["f3"] == 0.0


def test_normalization_degenerate():
    scores = {"f1": 3.0, "f2": 3.0, "f3": 3.0}
    norm = normalize_scores(scores)
    assert all(norm == 1.0)


def test_hadamard_lambda_zero_identity():
    """Mandatory unit test: lambda=0 must equal raw normalized ranking exactly."""
    scores = {"f1": 100.0, "f2": 50.0, "f3": 10.0, "f4": 1.0}
    weights = {"f1": 0.1, "f2": 0.2, "f3": 0.9, "f4": 1.0}  # Inverse weights
    fused = fuse(scores, weights, method="hadamard", strength=0.0)
    norm = normalize_scores(scores)
    
    # At lambda=0, scores must match norm exactly
    pd.testing.assert_series_equal(fused.scores, norm)
    assert fused.top_k(2) == ["f1", "f2"]


def test_hadamard_suppression_at_zero_weight():
    scores = {"f1": 100.0, "f2": 50.0, "f3": 0.0}
    weights = {"f1": 0.0, "f2": 1.0, "f3": 1.0}
    fused = fuse(scores, weights, method="hadamard", strength=1.0)
    assert fused.scores["f1"] == 0.0
    assert fused.scores["f2"] == 0.5
    assert fused.top_k(1) == ["f2"]


def test_rrf_formula():
    """RRF matches hand-computed example with k = 60."""
    scores = {"f1": 100.0, "f2": 50.0, "f3": 10.0}
    weights = {"f1": 0.2, "f2": 0.5, "f3": 1.0}
    # Math ranks: f1 -> 1, f2 -> 2, f3 -> 3
    # Clin ranks: f3 -> 1, f2 -> 2, f1 -> 3
    # At lambda = 1.0, k = 60:
    # f1: 1/(60 + 1) + 1/(60 + 3) = 1/61 + 1/63 = 0.01639344 + 0.01587302 = 0.03226646
    # f2: 1/(60 + 2) + 1/(60 + 2) = 2/62 = 0.03225806
    # f3: 1/(60 + 3) + 1/(60 + 1) = 1/63 + 1/61 = 0.03226646
    fused = fuse(scores, weights, method="rrf", strength=1.0, rrf_k=60)
    expected_f1 = (1.0 / 61) + (1.0 / 63)
    expected_f2 = (1.0 / 62) + (1.0 / 62)
    expected_f3 = (1.0 / 63) + (1.0 / 61)
    assert fused.scores["f1"] == pytest.approx(expected_f1, rel=1e-5)
    assert fused.scores["f2"] == pytest.approx(expected_f2, rel=1e-5)
    assert fused.scores["f3"] == pytest.approx(expected_f3, rel=1e-5)


def test_rrf_lambda_zero_identity():
    """Verify that RRF at lambda=0 reproduces baseline math Top-K exactly with Jaccard 1.0."""
    scores = {"f_b": 100.0, "f_a": 50.0, "f_c": 50.0, "f_d": 10.0}
    weights = {"f_b": 0.1, "f_a": 0.9, "f_c": 0.2, "f_d": 1.0}
    
    base_hadamard = fuse(scores, weights, method="hadamard", strength=0.0)
    rrf_zero = fuse(scores, weights, method="rrf", strength=0.0)
    
    # Both must produce identical Top-K ranking for all k
    for k in [1, 2, 3, 4]:
        base_top = base_hadamard.top_k(k)
        rrf_top = rrf_zero.top_k(k)
        assert base_top == rrf_top, f"Mismatch at k={k}: base={base_top}, rrf={rrf_top}"
        jaccard = len(set(base_top) & set(rrf_top)) / len(set(base_top) | set(rrf_top))
        assert jaccard == 1.0


def test_neutral_weights_identity():
    """All weights = 1.0 reproduces the baseline ranking under both Hadamard and RRF."""
    scores = {"feat_x": 80.0, "feat_y": 40.0, "feat_z": 20.0}
    neutral_w = {"feat_x": 1.0, "feat_y": 1.0, "feat_z": 1.0}
    
    base = fuse(scores, neutral_w, method="hadamard", strength=0.0)
    had_neutral = fuse(scores, neutral_w, method="hadamard", strength=1.0)
    rrf_zero = fuse(scores, neutral_w, method="rrf", strength=0.0)
    
    for k in [1, 2, 3]:
        assert had_neutral.top_k(k) == base.top_k(k)
        assert rrf_zero.top_k(k) == base.top_k(k)


def test_rank_tie_breaking_deterministic_and_permutation_invariant():
    """Tied features must be broken deterministically by feature name and invariant to column order."""
    scores_1 = pd.Series({"feat_b": 10.0, "feat_a": 5.0, "feat_c": 5.0, "feat_d": 1.0})[["feat_b", "feat_a", "feat_c", "feat_d"]]
    scores_2 = pd.Series({"feat_b": 10.0, "feat_a": 5.0, "feat_c": 5.0, "feat_d": 1.0})[["feat_c", "feat_a", "feat_b", "feat_d"]]
    weights = {"feat_a": 0.8, "feat_b": 0.5, "feat_c": 0.8, "feat_d": 0.2}

    res_1 = fuse(scores_1, weights, method="rrf", strength=1.0)
    res_2 = fuse(scores_2, weights, method="rrf", strength=1.0)

    # Top-K must be completely identical regardless of input order
    for k in [1, 2, 3, 4]:
        assert res_1.top_k(k) == res_2.top_k(k)
        assert res_1.top_k(k) == sorted(res_1.top_k(k), key=lambda x: (-res_1.scores[x], x))

