"""
Preprocessing script for CDC BRFSS Diabetes Health Indicators dataset.
Standardizes column names, checks types and missing values, creates clean parquet.
"""

import argparse
import re
import sys
from pathlib import Path
import pandas as pd


def to_snake_case(name: str) -> str:
    """Convert camelCase/PascalCase to snake_case."""
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def preprocess_diabetes(
    raw_path: str = "data/raw/diabetes_raw.csv",
    out_path: str = "data/processed/diabetes_clean.parquet",
    sample_size: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")

    print(f"Loading raw dataset from {raw_file}...")
    df = pd.read_csv(raw_file)
    print(f"Initial raw shape: {df.shape}")

    # Standardize column names to snake_case
    df.columns = [to_snake_case(c) for c in df.columns]
    print(f"Columns: {df.columns.tolist()}")

    # Target check
    target_col = "diabetes_binary"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Drop any nulls if present (CDC BRFSS is typically pre-cleaned, but enforce)
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with missing values.")

    # Cast integer columns to int64 or keep float if continuous
    # For BRFSS, most are ordinal/binary categories except BMI, MentHlth, PhysHlth, Age, Education, Income
    df[target_col] = df[target_col].astype(int)

    # Optional downsampling if specified (e.g. for fast protocol iteration / memory budget)
    if sample_size and sample_size < len(df):
        print(f"Subsampling to {sample_size} records (seed={seed}, stratified)...")
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[target_col],
            random_state=seed,
        )
        print(f"Subsampled shape: {df.shape}")

    # Ensure output dir exists
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving clean parquet to {out_file}...")
    df.to_parquet(out_file, index=False)
    print(f"Clean matrix saved successfully: {df.shape}")
    print("Class distribution:")
    print(df[target_col].value_counts(normalize=True))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Diabetes Dataset")
    parser.add_argument("--raw", default="data/raw/diabetes_raw.csv", help="Raw CSV path")
    parser.add_argument("--out", default="data/processed/diabetes_clean.parquet", help="Clean parquet path")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()
    preprocess_diabetes(raw_path=args.raw, out_path=args.out, sample_size=args.sample, seed=args.seed)
