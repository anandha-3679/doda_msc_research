"""
Preprocessing script for Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
Standardizes column names to snake_case matching configs/weights/breast_cancer_weights.json,
drops patient ID, maps diagnosis ('M' -> 1, 'B' -> 0), and outputs clean parquet matrix.
"""

import argparse
from pathlib import Path
import pandas as pd

# The 10 base nuclear morphology metrics computed from FNA digitized images
BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

# Attribute ordering in WDBC data: ID, diagnosis, 10 mean, 10 SE (error), 10 worst
FEATURE_NAMES = (
    [f"mean_{f}" for f in BASE_FEATURES]
    + [f"{f}_error" for f in BASE_FEATURES]
    + [f"worst_{f}" for f in BASE_FEATURES]
)

ID_COL = "id"
TARGET_COL = "diagnosis"
RAW_COLUMNS = [ID_COL, TARGET_COL] + FEATURE_NAMES


def preprocess_breast_cancer(
    raw_path: str = "data/raw/breast_cancer_raw/wdbc.data",
    out_path: str = "data/processed/breast_cancer_clean.parquet",
) -> pd.DataFrame:
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")

    print(f"Loading raw WDBC dataset from {raw_file}...")
    df = pd.read_csv(raw_file, header=None, names=RAW_COLUMNS)
    print(f"Initial raw shape: {df.shape}")

    # Verify absence of missing values
    missing_count = df.isna().sum().sum()
    print(f"Total missing values: {missing_count}")

    # Drop ID column to avoid data leakage / spurious patterns
    df = df.drop(columns=[ID_COL])

    # Target binarization: M (Malignant) -> 1, B (Benign) -> 0
    df[TARGET_COL] = df[TARGET_COL].map({"M": 1, "B": 0}).astype(int)

    # Ensure all feature columns are numeric float64
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving clean parquet to {out_file}...")
    df.to_parquet(out_file, index=False)
    print(f"Clean matrix saved successfully: {df.shape}")
    print("Class distribution (0: Benign, 1: Malignant):")
    print(df[TARGET_COL].value_counts(normalize=True))
    print(df[TARGET_COL].value_counts())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WDBC Breast Cancer Dataset")
    parser.add_argument("--raw", default="data/raw/breast_cancer_raw/wdbc.data", help="Raw dataset path")
    parser.add_argument("--out", default="data/processed/breast_cancer_clean.parquet", help="Clean parquet output path")

    args = parser.parse_args()
    preprocess_breast_cancer(raw_path=args.raw, out_path=args.out)
