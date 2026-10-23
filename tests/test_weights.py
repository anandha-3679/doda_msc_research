"""
Unit tests for clinical weight validation.
"""

import pytest
from doda.knowledge.validator import load_and_validate_weights, validate_provenance, WeightValidationError


def test_diabetes_weights_validation():
    features = [
        "high_bp", "high_chol", "chol_check", "bmi", "smoker", "stroke",
        "heart_diseaseor_attack", "phys_activity", "fruits", "veggies",
        "hvy_alcohol_consump", "any_healthcare", "no_docbc_cost", "gen_hlth",
        "ment_hlth", "phys_hlth", "diff_walk", "sex", "age", "education", "income"
    ]
    weights, meta = load_and_validate_weights(
        "configs/weights/diabetes_weights.json",
        feature_names=features
    )
    assert len(weights) == len(features)
    assert weights["high_bp"] == 1.0
    assert weights["sex"] == 0.2
    assert validate_provenance("configs/weights/provenance.csv", list(weights.keys()))


def test_invalid_weight_bounds(tmp_path):
    p = tmp_path / "bad_weights.json"
    p.write_text('{"weights": {"bmi": 1.5}}')
    with pytest.raises(WeightValidationError):
        load_and_validate_weights(str(p))
