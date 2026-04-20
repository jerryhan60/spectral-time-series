"""
Pairwise statistical significance between ablation variants.

Tests whether the differences between MS46 and each ablation are significant.
Uses Wilcoxon signed-rank test on paired per-config MASE values.
"""

import numpy as np
import pandas as pd
from scipy import stats
import glob, os

RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
TABLEDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/tables"

def find_csv_by_mase(target_mase, tol=0.01):
    """Find CSV with geo mean closest to target."""
    all_csvs = glob.glob(os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_*.csv"))
    best_path, best_diff = None, 999
    for csv_path in all_csvs:
        df = pd.read_csv(csv_path)
        mase_col = [c for c in df.columns if 'MASE' in c and '0.5' in c]
        if not mase_col:
            mase_col = [c for c in df.columns if 'MASE' in c]
        if not mase_col:
            continue
        mase = df[mase_col[0]].dropna()
        mase = mase[mase > 0]
        if len(mase) == 97:
            gm = np.exp(np.mean(np.log(mase)))
            diff = abs(gm - target_mase)
            if diff < best_diff and diff < tol:
                best_diff = diff
                best_path = csv_path
    return best_path

# Known MASE values for each method
methods = {
    'Baseline': 1.2421,
    'MS46': 1.1675,
    'Duplicate': 1.2342,
    'Learned 4-tap': 1.2351,
    'Learned 16-tap': 1.2775,
    'Zero hints': 1.3201,
    'Random hints': 1.2806,
}

# Load all available CSVs
method_data = {}
for name, target_mase in methods.items():
    path = find_csv_by_mase(target_mase)
    if path:
        df = pd.read_csv(path)
        mase_col = [c for c in df.columns if 'MASE' in c and '0.5' in c]
        if not mase_col:
            mase_col = [c for c in df.columns if 'MASE' in c]
        mase = df[mase_col[0]].values
        method_data[name] = mase
        gm = np.exp(np.mean(np.log(mase[mase > 0])))
        print(f"Loaded {name}: n={len(mase)}, geoMASE={gm:.4f} (target={target_mase:.4f})")
    else:
        print(f"SKIP {name}: no matching CSV found")

print("\n=== Pairwise Wilcoxon Signed-Rank Tests ===")
print(f"{'Method A':20s} {'Method B':20s} {'p-value':>10s} {'Cohen d':>10s} {'Win A':>8s} {'Sig?':>6s}")
print("-" * 80)

method_names = list(method_data.keys())
results_rows = []

for i in range(len(method_names)):
    for j in range(i+1, len(method_names)):
        name_a = method_names[i]
        name_b = method_names[j]
        mase_a = method_data[name_a]
        mase_b = method_data[name_b]

        # Paired Wilcoxon
        try:
            stat, p_val = stats.wilcoxon(mase_a, mase_b, alternative='two-sided')
        except:
            p_val = 1.0

        # Cohen's d on log ratios
        log_ratio = np.log(mase_a) - np.log(mase_b)
        cohens_d = np.mean(log_ratio) / np.std(log_ratio) if np.std(log_ratio) > 0 else 0

        # Win rate
        wins_a = np.sum(mase_a < mase_b)
        wins_b = np.sum(mase_b < mase_a)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        print(f"{name_a:20s} {name_b:20s} {p_val:10.2e} {cohens_d:10.3f} {wins_a:>3d}/{len(mase_a):<4d} {sig:>6s}")
        results_rows.append({
            'Method A': name_a, 'Method B': name_b,
            'p_value': p_val, 'cohens_d': cohens_d,
            'wins_a': wins_a, 'wins_b': wins_b, 'sig': sig
        })

# Generate LaTeX table for key comparisons
print("\n=== Key Comparisons (LaTeX) ===")
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Pairwise statistical comparisons (Wilcoxon signed-rank test, 97 paired configs). Cohen's $d$ computed on log MASE ratios.}")
lines.append(r"\label{tab:significance}")
lines.append(r"\begin{tabular}{llcrr}")
lines.append(r"\toprule")
lines.append(r"Method A & Method B & $p$-value & Cohen's $d$ & Win rate \\")
lines.append(r"\midrule")

for row in results_rows:
    p_str = f"{row['p_value']:.1e}" if row['p_value'] < 0.001 else f"{row['p_value']:.3f}"
    win_str = f"{row['wins_a']}/{row['wins_a'] + row['wins_b']}"
    sig_str = row['sig']
    lines.append(f"{row['Method A']} & {row['Method B']} & {p_str}{sig_str} & {row['cohens_d']:.3f} & {win_str} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

latex = '\n'.join(lines)
print(latex)
with open(f'{TABLEDIR}/table_significance.tex', 'w') as f:
    f.write(latex)
print(f"\nSaved table_significance.tex")
