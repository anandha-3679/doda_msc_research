"""
Preprocessing script for canonical UCI Chronic Kidney Disease (CKD) dataset
(Apollo Hospitals, 400 patients).
Parses ARFF data, standardizes column names, encodes binary indicators,
handles missing value sentinel '?', and outputs clean parquet matrix at
data/processed/ckd_uci_clean.parquet.
"""

import argparse
from pathlib import Path
import numpy as np
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
ALL_COLUMNS = FEATURE_NAMES + [TARGET_COL]

BINARY_MAPS = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
    "class": {"notckd": 0, "ckd": 1},
}


def preprocess_ckd_uci(
    raw_path: str = "data/raw/Chronic_Kidney_Disease/chronic_kidney_disease_full.arff",
    out_path: str = "data/processed/ckd_uci_clean.parquet",
) -> pd.DataFrame:
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw UCI CKD file not found: {raw_file}")

    print(f"Loading canonical UCI CKD dataset from {raw_file}...")
    with open(raw_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    try:
        data_idx = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data") + 1
    except StopIteration:
        raise ValueError("Could not find @data marker in ARFF file.")

    rows = []
    for line in lines[data_idx:]:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("%"):
            continue
        # Clean trailing commas
        if cleaned.endswith(","):
            cleaned = cleaned[:-1]
        # Fix known formatting typo in line 369 where double comma appears
        if "no,,no,no,good" in cleaned:
            cleaned = cleaned.replace("no,,no,no,good", "no,no,no,good")
        parts = [p.strip() for p in cleaned.split(",")]
        rows.append(parts)

    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    print(f"Initial raw parsed shape: {df.shape}")

    # Standardize missing value sentinels ('?' -> NaN)
    df = df.replace("?", np.nan)

    # Encode nominal binary variables
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping)

    # Convert all columns to numeric
    for col in ALL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate target column has no missing values and is 0/1 integer
    if df[TARGET_COL].isna().any():
        raise ValueError("Target column contains missing values!")
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Reorder columns consistently
    df = df[FEATURE_NAMES + [TARGET_COL]].copy()

    # Save clean parquet
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving clean parquet to {out_file}...")
    df.to_parquet(out_file, index=False)
    print(f"Clean matrix saved successfully: {df.shape}")
    print("Target distribution (0: notckd, 1: ckd):")
    print(df[TARGET_COL].value_counts(normalize=True))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Canonical UCI Chronic Kidney Disease (CKD) Dataset")
    parser.add_argument(
        "--raw",
        default="data/raw/Chronic_Kidney_Disease/chronic_kidney_disease_full.arff",
        help="Raw ARFF dataset path",
    )
    parser.add_argument(
        "--out",
        default="data/processed/ckd_uci_clean.parquet",
        help="Clean parquet output path",
    )

    args = parser.parse_args()
    preprocess_ckd_uci(raw_path=args.raw, out_path=args.out)
