"""
Clinical Weight Validator and Loader for DODA.
Validates weight JSON against strict schema and matching features.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


class WeightValidationError(Exception):
    """Raised when clinical weight specification violates schema or bounds."""
    pass


def load_and_validate_weights(
    weights_path: str,
    feature_names: Optional[List[str]] = None,
    default_weight: float = 1.0,
) -> Tuple[Dict[str, float], dict]:
    """
    Load weights from JSON file and validate against feature list and range bounds.
    
    Rules:
    - Weights must be strictly in [0.0, 1.0].
    - Unknown keys raise WeightValidationError.
    - Features not in weights file default to default_weight (1.0).
    """
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Weights file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("_meta", {})
    weights_dict = data.get("weights", {})

    # Validate weight values in [0.0, 1.0]
    for feat, w in weights_dict.items():
        if not isinstance(w, (int, float)):
            raise WeightValidationError(f"Weight for feature '{feat}' must be numeric, got {type(w)}")
        if not (0.0 <= w <= 1.0):
            raise WeightValidationError(f"Weight for feature '{feat}' must be in [0.0, 1.0], got {w}")

    # Validate against known feature names if provided
    if feature_names is not None:
        feature_set = set(feature_names)
        unknown_features = set(weights_dict.keys()) - feature_set
        if unknown_features:
            raise WeightValidationError(
                f"Unknown feature names in weights file not found in dataset: {unknown_features}"
            )

        # Build complete map with defaults
        full_weights = {}
        for feat in feature_names:
            full_weights[feat] = float(weights_dict.get(feat, meta.get("default_weight", default_weight)))
    else:
        full_weights = {k: float(v) for k, v in weights_dict.items()}

    return full_weights, meta


def validate_provenance(provenance_path: str, weight_keys: List[str]) -> bool:
    """Ensure every specified weight has a corresponding entry in provenance.csv."""
    p = Path(provenance_path)
    if not p.exists():
        raise FileNotFoundError(f"Provenance file not found: {p}")
    df = pd.read_csv(p)
    documented_features = set(df["feature"].tolist())
    missing = set(weight_keys) - documented_features
    if missing:
        raise WeightValidationError(f"Features missing from provenance table: {missing}")
    return True
