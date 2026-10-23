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


def test_rrf_fusion():
    scores = {"f1": 100.0, "f2": 50.0, "f3": 10.0}
    weights = {"f1": 0.2, "f2": 0.5, "f3": 1.0}
    fused = fuse(scores, weights, method="rrf", strength=1.0)
    assert len(fused.top_k(3)) == 3
