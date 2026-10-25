"""
Sanity tests per dataset to verify expected hallmark biomarker differences across classes.
- Diabetes: high_bp, bmi, gen_hlth, age should have higher mean in diabetic patients (class 1 > class 0).
- Heart Disease: ca, thal, cp, oldpeak should be higher in heart disease patients (class 1 > class 0).
- Breast Cancer: worst_concave_points, worst_perimeter, mean_radius should be higher in malignant tumors (class 1 > class 0).
- CKD (Quarantined audit check): Documented lack of expected biomarker separation in BD-KDD raw file (sc ~ 7.4 vs 7.6, al ~ 2.03 vs 2.02).
"""

from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_diabetes_sanity():
    p = ROOT_DIR / "data/processed/diabetes_clean.parquet"
    assert p.exists(), "Clean diabetes parquet matrix must exist"
    df = pd.read_parquet(p)
    
    # Positive class (1 = Diabetes) vs Negative class (0 = No Diabetes)
    mean_bp_0 = df[df["diabetes_binary"] == 0]["high_bp"].mean()
    mean_bp_1 = df[df["diabetes_binary"] == 1]["high_bp"].mean()
    assert mean_bp_1 > mean_bp_0, "Diabetic patients should have higher prevalence of high_bp"
    assert mean_bp_1 > 0.70 and mean_bp_0 < 0.40

    mean_bmi_0 = df[df["diabetes_binary"] == 0]["bmi"].mean()
    mean_bmi_1 = df[df["diabetes_binary"] == 1]["bmi"].mean()
    assert mean_bmi_1 > mean_bmi_0, "Diabetic patients should have higher BMI"

    mean_gen_0 = df[df["diabetes_binary"] == 0]["gen_hlth"].mean()
    mean_gen_1 = df[df["diabetes_binary"] == 1]["gen_hlth"].mean()
    assert mean_gen_1 > mean_gen_0, "Diabetic patients should report worse general health"


def test_heartdisease_sanity():
    p = ROOT_DIR / "data/processed/heartdisease_clean.parquet"
    assert p.exists(), "Clean heart disease parquet matrix must exist"
    df = pd.read_parquet(p)
    
    # Target 1 = Disease (stenosis >= 50%), Target 0 = No Disease
    mean_ca_0 = df[df["target"] == 0]["ca"].mean()
    mean_ca_1 = df[df["target"] == 1]["ca"].mean()
    assert mean_ca_1 > mean_ca_0, "Heart disease patients should have significantly more fluoroscopy-stained vessels"
    assert mean_ca_1 > 1.0 and mean_ca_0 < 0.4

    mean_oldpeak_0 = df[df["target"] == 0]["oldpeak"].mean()
    mean_oldpeak_1 = df[df["target"] == 1]["oldpeak"].mean()
    assert mean_oldpeak_1 > mean_oldpeak_0, "Heart disease patients should have higher ST depression"

    mean_cp_0 = df[df["target"] == 0]["cp"].mean()
    mean_cp_1 = df[df["target"] == 1]["cp"].mean()
    assert mean_cp_1 > mean_cp_0, "Heart disease patients should have higher chest pain classification score"


def test_breast_cancer_sanity():
    p = ROOT_DIR / "data/processed/breast_cancer_clean.parquet"
    assert p.exists(), "Clean breast cancer parquet matrix must exist"
    df = pd.read_parquet(p)
    
    # Diagnosis 1 = Malignant, Diagnosis 0 = Benign
    mean_conc_0 = df[df["diagnosis"] == 0]["worst_concave_points"].mean()
    mean_conc_1 = df[df["diagnosis"] == 1]["worst_concave_points"].mean()
    assert mean_conc_1 > mean_conc_0, "Malignant tumors should have significantly higher worst concave points"
    assert mean_conc_1 > 2.0 * mean_conc_0

    mean_perim_0 = df[df["diagnosis"] == 0]["worst_perimeter"].mean()
    mean_perim_1 = df[df["diagnosis"] == 1]["worst_perimeter"].mean()
    assert mean_perim_1 > mean_perim_0, "Malignant tumors should have substantially larger worst perimeter"

    mean_rad_0 = df[df["diagnosis"] == 0]["mean_radius"].mean()
    mean_rad_1 = df[df["diagnosis"] == 1]["mean_radius"].mean()
    assert mean_rad_1 > mean_rad_0, "Malignant tumors should have larger mean cell nucleus radius"


def test_ckd_uci_sanity():
    p = ROOT_DIR / "data/processed/ckd_uci_clean.parquet"
    assert p.exists(), "Clean UCI CKD parquet matrix must exist"
    df = pd.read_parquet(p)
    
    # Target 1 = CKD, Target 0 = notckd
    mean_sc_0 = df[df["class"] == 0]["sc"].mean()
    mean_sc_1 = df[df["class"] == 1]["sc"].mean()
    assert mean_sc_1 > mean_sc_0 + 2.0, "CKD patients must have substantially higher serum creatinine"

    mean_al_0 = df[df["class"] == 0]["al"].mean()
    mean_al_1 = df[df["class"] == 1]["al"].mean()
    assert mean_al_1 > mean_al_0 + 1.0, "CKD patients must have higher albuminuria"

    mean_hemo_0 = df[df["class"] == 0]["hemo"].mean()
    mean_hemo_1 = df[df["class"] == 1]["hemo"].mean()
    assert mean_hemo_0 > mean_hemo_1 + 3.0, "Healthy controls must have significantly higher hemoglobin"


def test_quarantined_ckd_audit():
    p = ROOT_DIR / "data/quarantine/ckd_raw.csv"
    assert p.exists(), "Quarantined CKD raw dataset must exist in data/quarantine"
    df = pd.read_csv(p)
    
    # Audit verification: verify near-zero separation between Class 0 and Class 1 for Serum Creatinine & Albumin
    mean_sc_0 = df[df["Class"] == 0]["Sc"].mean()
    mean_sc_1 = df[df["Class"] == 1]["Sc"].mean()
    diff_sc = abs(mean_sc_1 - mean_sc_0)
    assert diff_sc < 0.5, f"Serum creatinine shows suspicious lack of difference ({mean_sc_0:.2f} vs {mean_sc_1:.2f}) in BD-KDD"

    mean_al_0 = df[df["Class"] == 0]["Al"].mean()
    mean_al_1 = df[df["Class"] == 1]["Al"].mean()
    diff_al = abs(mean_al_1 - mean_al_0)
    assert diff_al < 0.1, f"Albumin shows suspicious lack of difference ({mean_al_0:.2f} vs {mean_al_1:.2f}) in BD-KDD"


def test_quarantine_pipeline_isolation():
    """
    Verify that the suspect CKD dataset cannot silently enter the active primary benchmark:
    1. Active dataset config directory (configs/datasets/) must only contain verified cohorts.
    2. No unquarantined CKD data may exist in data/raw/ or data/processed/.
    3. The quarantined raw file exists with preserved SHA-256 checksum.
    """
    import hashlib

    # 1. Config isolation: only verified cohorts allowed, ckd.yaml must remain quarantined
    active_configs_dir = ROOT_DIR / "configs/datasets"
    active_yaml_files = [f.name for f in active_configs_dir.glob("*.yaml")]
    assert "ckd.yaml" not in active_yaml_files, "ckd.yaml must not exist in active configs/datasets/"
    assert set(active_yaml_files) == {"diabetes.yaml", "heartdisease.yaml", "breast_cancer.yaml", "ckd_uci.yaml"}

    # 2. Data path isolation: BD-KDD file names must not leak
    assert not (ROOT_DIR / "data/raw/ckd_raw.csv").exists(), "ckd_raw.csv must not exist in data/raw/"
    assert not (ROOT_DIR / "data/processed/ckd_clean.parquet").exists(), "ckd_clean.parquet must not exist in data/processed/"

    # 3. Quarantine preservation and SHA-256 integrity
    quarantined_raw = ROOT_DIR / "data/quarantine/ckd_raw.csv"
    assert quarantined_raw.exists(), "ckd_raw.csv must be safely preserved in data/quarantine/"
    
    with open(quarantined_raw, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    
    expected_hash = "cb9d92f77f20859edaf73ba9236d8fed814e854016ab2eaf83e13ddba45bfaeb"
    assert sha256_hash == expected_hash, f"Quarantined raw SHA-256 mismatch: {sha256_hash}"

    # 4. Quarantined config exists
    quarantined_cfg = ROOT_DIR / "data/quarantine/ckd.yaml"
    assert quarantined_cfg.exists(), "ckd.yaml must be preserved in data/quarantine/"

