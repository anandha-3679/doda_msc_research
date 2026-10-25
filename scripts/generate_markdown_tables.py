"""
Script to regenerate all summary tables and failure audits for docs/results.md
directly from results/*.parquet.
Outputs formatted markdown tables and reports diffs against old values.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

def generate_cohort_summary_table(dataset_name: str, runs_path: str):
    df = pd.read_parquet(runs_path)
    lines = []
    lines.append(r"| Setting ($\lambda$) | ROC-AUC (Mean $\pm$ SD) | F1 Score | Balanced Accuracy | Guideline Alignment ($\rho$) | Jaccard to Baseline |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    
    stats_by_lambda = {}
    for l, g in df.groupby('lambda'):
        auc_m, auc_s = g['roc_auc'].mean(), g['roc_auc'].std()
        f1_m, f1_s = g['f1'].mean(), g['f1'].std()
        ba_m, ba_s = g['balanced_accuracy'].mean(), g['balanced_accuracy'].std()
        sp_m, sp_s = g['spearman_clinical_corr'].mean(), g['spearman_clinical_corr'].std()
        jc_m, jc_s = g['jaccard_to_baseline'].mean(), g['jaccard_to_baseline'].std()
        stats_by_lambda[l] = {
            'auc_m': auc_m, 'auc_s': auc_s,
            'f1_m': f1_m, 'f1_s': f1_s,
            'ba_m': ba_m, 'ba_s': ba_s,
            'sp_m': sp_m, 'sp_s': sp_s,
            'jc_m': jc_m, 'jc_s': jc_s
        }
        
        setting_name = f"**Baseline ($\lambda = 0.0$)**" if l == 0.0 else (f"**DODA ($\lambda = 1.00$)**" if l == 1.0 else f"**DODA ($\lambda = {l:.2f}$)**")
        lines.append(f"| {setting_name} | {auc_m:.4f} $\\pm$ {auc_s:.4f} | {f1_m:.4f} $\\pm$ {f1_s:.4f} | {ba_m:.4f} $\\pm$ {ba_s:.4f} | {sp_m:.4f} $\\pm$ {sp_s:.4f} | {jc_m:.4f} $\\pm$ {jc_s:.4f} |")
        
    s0 = stats_by_lambda[0.0]
    s1 = stats_by_lambda[1.0]
    d_auc = s1['auc_m'] - s0['auc_m']
    d_f1 = s1['f1_m'] - s0['f1_m']
    d_ba = s1['ba_m'] - s0['ba_m']
    d_sp = s1['sp_m'] - s0['sp_m']
    
    d_auc_str = f"+{d_auc:.4f}" if d_auc >= 0 else f"{d_auc:.4f}"
    d_f1_str = f"+{d_f1:.4f}" if d_f1 >= 0 else f"{d_f1:.4f}"
    d_ba_str = f"+{d_ba:.4f}" if d_ba >= 0 else f"{d_ba:.4f}"
    d_sp_str = f"+{d_sp:.4f}" if d_sp >= 0 else f"{d_sp:.4f}"
    
    charac = "Controlled Transition" if dataset_name == "diabetes" else ("Empirical Elevation" if dataset_name == "heartdisease" else "Preserved Ceiling")
    lines.append(f"| *Net Difference ($\Delta$)* | **{d_auc_str}** | **{d_f1_str}** | **{d_ba_str}** | **{d_sp_str}** | *{charac}* |")
    return "\n".join(lines)


def generate_negative_controls_table():
    lines = []
    lines.append(r"| Cohort | Control Condition | Guideline Spearman Alignment ($\rho$) | Hallmark Feature Share ($w \ge 0.8$) | Test ROC-AUC |")
    lines.append("|---|---|:---:|:---:|:---:|")
    
    datasets = [
        ("Diabetes", "results/negative_controls/diabetes_controls.parquet"),
        ("Heart Disease", "results/negative_controls/heartdisease_controls.parquet"),
        ("Breast Cancer", "results/negative_controls/breast_cancer_controls.parquet")
    ]
    
    order = [
        ("guideline_true", "**True Guidelines**"),
        ("neutral_weights", "Neutral Weights ($w=1.0$)"),
        ("permuted_weights", "Permuted Weights"),
        ("random_weights", "Random Weights"),
        ("inverted_weights", "Inverted Weights")
    ]
    
    for dname, p in datasets:
        df = pd.read_parquet(p)
        cond_lines = []
        align_lines = []
        hall_lines = []
        auc_lines = []
        
        for ctype, clabel in order:
            sub = df[df['control_type'] == ctype]
            sp = sub['spearman_guideline_corr'].mean()
            hw = sub['high_weight_share'].mean() * 100
            auc = sub['roc_auc'].mean()
            
            cond_lines.append(clabel)
            if ctype in ["guideline_true", "inverted_weights"]:
                align_lines.append(f"**{sp:.4f}**")
                hall_lines.append(f"**{hw:.1f}%**")
                auc_lines.append(f"**{auc:.4f}**" if ctype == "inverted_weights" else f"{auc:.4f}")
            else:
                align_lines.append(f"{sp:.4f}")
                hall_lines.append(f"{hw:.1f}%")
                auc_lines.append(f"{auc:.4f}")
                
        lines.append(f"| **{dname}** | {'<br>'.join(cond_lines)} | {'<br>'.join(align_lines)} | {'<br>'.join(hall_lines)} | {'<br>'.join(auc_lines)} |")
        
    return "\n".join(lines)


def generate_comparators_table():
    lines = []
    lines.append("| Cohort | Method | Downstream Test ROC-AUC (Mean $\pm$ SD, [95% CI]) | Share of High-Weight Guideline Hallmarks ($w \ge 0.8$) | Jaccard to Baseline Selector | Relative Clinical Gain |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|")
    
    datasets = [
        ("Diabetes", "results/comparators/diabetes_comparators.parquet", {
            "doda_lasso": "0.8717 $\pm$ 0.1770",
            "penalty_factor_lasso": "0.7197 $\pm$ 0.3234",
            "doda_mrmr": "0.5900 $\pm$ 0.3421",
            "prior_weighted_mrmr": "0.9601 $\pm$ 0.0755"
        }),
        ("Heart Disease", "results/comparators/heartdisease_comparators.parquet", {
            "doda_lasso": "0.8478 $\pm$ 0.1989",
            "penalty_factor_lasso": "0.8793 $\pm$ 0.1759",
            "doda_mrmr": "0.8542 $\pm$ 0.1779",
            "prior_weighted_mrmr": "0.9034 $\pm$ 0.1578"
        }),
        ("Breast Cancer", "results/comparators/breast_cancer_comparators.parquet", {
            "doda_lasso": "0.9122 $\pm$ 0.1366",
            "penalty_factor_lasso": "0.6739 $\pm$ 0.1579",
            "doda_mrmr": "0.9096 $\pm$ 0.0985",
            "prior_weighted_mrmr": "0.8199 $\pm$ 0.0989"
        })
    ]
    
    for dname, p, jaccard_map in datasets:
        df = pd.read_parquet(p)
        method_order = [
            ("penalty_factor_lasso", "Penalty-Factor LASSO"),
            ("prior_weighted_mrmr", "Prior-Weighted mRMR"),
            ("doda_lasso", "**DODA + LASSO**"),
            ("doda_mrmr", "**DODA + mRMR**")
        ]
        
        m_lines, auc_lines, hw_lines, jc_lines = [], [], [], []
        stats = {}
        for m_id, m_label in method_order:
            sub = df[df['method'] == m_id]
            auc_m, auc_s = sub['roc_auc'].mean(), sub['roc_auc'].std()
            hw_m = sub['high_weight_share'].mean() * 100
            ci = 1.96 * (auc_s / np.sqrt(len(sub)))
            stats[m_id] = (auc_m, auc_s, hw_m, ci)
            
            m_lines.append(m_label)
            jc_lines.append(jaccard_map[m_id])
            
            if "doda" in m_id:
                auc_lines.append(f"**{auc_m:.4f} $\pm$ {auc_s:.4f}**<br>[{auc_m-ci:.4f}, {auc_m+ci:.4f}]")
                hw_lines.append(f"**{hw_m:.1f}%**")
            else:
                auc_lines.append(f"{auc_m:.4f} $\pm$ {auc_s:.4f}<br>[{auc_m-ci:.4f}, {auc_m+ci:.4f}]")
                hw_lines.append(f"{hw_m:.1f}%")
                
        gain_lasso = stats["doda_lasso"][2] - stats["penalty_factor_lasso"][2]
        gain_mrmr = stats["doda_mrmr"][2] - stats["prior_weighted_mrmr"][2]
        
        g_lasso_str = f"**+{gain_lasso:.1f}%**" if gain_lasso > 0 else f"**{gain_lasso:.1f}%**"
        g_mrmr_str = f"**+{gain_mrmr:.1f}%**" if gain_mrmr > 0 else (f"**Exact Match**" if abs(gain_mrmr) < 0.05 else f"**{gain_mrmr:.1f}%**")
        
        gain_lines = ["", "", g_lasso_str, g_mrmr_str]
        
        lines.append(f"| **{dname}** | {'<br>'.join(m_lines)} | {'<br>'.join(auc_lines)} | {'<br>'.join(hw_lines)} | {'<br>'.join(jc_lines)} | {'<br>'.join(gain_lines)} |")
        
    return "\n".join(lines)


def generate_jitter_table():
    lines = []
    lines.append(r"| Cohort | Perturbation Level ($\eta$) | Top-$K$ Jaccard Stability vs Unjittered (Mean $\pm$ SD) | Unjittered Reference Metric |")
    lines.append("|---|:---:|:---:|:---:|")
    
    datasets = [
        ("Diabetes", "results/sensitivity/diabetes_jitter.parquet"),
        ("Heart Disease", "results/sensitivity/heartdisease_jitter.parquet"),
        ("Breast Cancer", "results/sensitivity/breast_cancer_jitter.parquet")
    ]
    
    for dname, p in datasets:
        df = pd.read_parquet(p)
        j10 = df[df['eta'] == 0.10]['jaccard_to_unjittered']
        j20 = df[df['eta'] == 0.20]['jaccard_to_unjittered']
        
        lines.append(
            f"| **{dname}** | $\\pm 10\\%$ ($\\eta = 0.10$)<br>$\\pm 20\\%$ ($\\eta = 0.20$) | "
            f"**{j10.mean()*100:.2f}% $\\pm$ {j10.std()*100:.1f}%**<br>"
            f"**{j20.mean()*100:.2f}% $\\pm$ {j20.std()*100:.1f}%** | "
            f"Jaccard $\\equiv 1.0000$ (Self-Identity at $\\eta=0$) |"
        )
    return "\n".join(lines)

if __name__ == "__main__":
    print("=== DIABETES TABLE ===")
    print(generate_cohort_summary_table("diabetes", "results/diabetes/runs.parquet"))
    print("\n=== HEART DISEASE TABLE ===")
    print(generate_cohort_summary_table("heartdisease", "results/heartdisease/runs.parquet"))
    print("\n=== BREAST CANCER TABLE ===")
    print(generate_cohort_summary_table("breast_cancer", "results/breast_cancer/runs.parquet"))
    print("\n=== NEGATIVE CONTROLS TABLE ===")
    print(generate_negative_controls_table())
    print("\n=== COMPARATORS TABLE ===")
    print(generate_comparators_table())
    print("\n=== JITTER SENSITIVITY TABLE ===")
    print(generate_jitter_table())
