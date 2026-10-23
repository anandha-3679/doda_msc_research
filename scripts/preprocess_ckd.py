"""
Preprocessing script for Chronic Kidney Disease (CKD) dataset.
Standardizes column names into snake_case, drops identifiers ('sl_no'),
validates target variable ('class' in {0, 1}), and outputs clean parquet matrix
at data/processed/ckd_clean.parquet.
"""

import argparse
from pathlib import Path
import pandas as pd

FEATURE_NAMES = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]
TARGET_COL = "class"


def preprocess_ckd(
    raw_path: str = "data/raw/ckd_raw.csv",
    out_path: str = "data/processed/ckd_clean.parquet",
) -> pd.DataFrame:
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")

    print(f"Loading raw CKD dataset from {raw_file}...")
    df = pd.read_csv(raw_file)
    print(f"Initial raw shape: {df.shape}")

    # Standardize column names to lower case
    df.columns = [c.strip().lower().replace(" ", "_").replace(".", "") for c in df.columns]

    # Drop identifier column if present (e.g. sl_no or sl_no_)
    id_cols = [c for c in df.columns if "sl" in c or "id" in c]
    if id_cols:
        print(f"Dropping identifier columns: {id_cols}")
        df = df.drop(columns=id_cols)

    # Validate target column exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns: {list(df.columns)}")

    # Ensure target is integer 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Ensure all feature columns exist and are numeric
    for col in FEATURE_NAMES:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' missing from CKD dataset.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Order columns consistently: features followed by target
    df = df[FEATURE_NAMES + [TARGET_COL]].copy()

    # Save clean parquet
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving clean parquet to {out_file}...")
    df.to_parquet(out_file, index=False)
    print(f"Clean matrix saved successfully: {df.shape}")
    print("Target distribution (0: non-CKD / control, 1: CKD):")
    print(df[TARGET_COL].value_counts(normalize=True))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Chronic Kidney Disease (CKD) Dataset")
    parser.add_argument("--raw", default="data/raw/ckd_raw.csv", help="Raw dataset path")
    parser.add_argument("--out", default="data/processed/ckd_clean.parquet", help="Clean parquet output path")

    args = parser.parse_args()
    preprocess_ckd(raw_path=args.raw, out_path=args.out)
