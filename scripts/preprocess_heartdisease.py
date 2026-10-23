"""
Preprocessing script for UCI Heart Disease (Cleveland) dataset.
Standardizes column names, handles missing values ('?'), binarizes the target,
and outputs clean parquet matrix at data/processed/heartdisease_clean.parquet.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Standard 14 Cleveland Heart Disease attributes:
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COL = "target"
ALL_COLUMNS = FEATURE_NAMES + [TARGET_COL]


def preprocess_heartdisease(
    raw_path: str = "data/raw/heartdisease_raw.data",
    out_path: str = "data/processed/heartdisease_clean.parquet",
    drop_missing: bool = False,
) -> pd.DataFrame:
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")

    print(f"Loading raw Cleveland heart disease dataset from {raw_file}...")
    df = pd.read_csv(raw_file, header=None, names=ALL_COLUMNS, na_values="?")
    print(f"Initial raw shape: {df.shape}")

    # Check missing values
    missing_counts = df.isna().sum()
    if missing_counts.sum() > 0:
        print("Missing values per column:")
        print(missing_counts[missing_counts > 0])
        if drop_missing:
            df = df.dropna().reset_index(drop=True)
            print(f"Dropped rows with missing values. New shape: {df.shape}")

    # Target binarization: 0 = absence, 1,2,3,4 = presence of heart disease
    # Standard clinical/benchmark definition (>= 50% diameter narrowing)
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    # Convert numeric types
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # If not dropping missing, in-fold SimpleImputer will handle them during CV,
    # but we ensure parquet format is written clean
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving clean parquet to {out_file}...")
    df.to_parquet(out_file, index=False)
    print(f"Clean matrix saved successfully: {df.shape}")
    print("Target distribution (0: absence, 1: presence):")
    print(df[TARGET_COL].value_counts(normalize=True))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UCI Heart Disease Dataset")
    parser.add_argument("--raw", default="data/raw/heartdisease_raw.data", help="Raw dataset path")
    parser.add_argument("--out", default="data/processed/heartdisease_clean.parquet", help="Clean parquet output path")
    parser.add_argument("--drop-missing", action="store_true", help="Drop missing values statically instead of in-fold")

    args = parser.parse_args()
    preprocess_heartdisease(raw_path=args.raw, out_path=args.out, drop_missing=args.drop_missing)
