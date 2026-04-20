"""
Main result figure for the paper.

Combines: (a) improvement distribution, (b) horizon scaling, (c) key ablation.
This is the "money figure" that tells the complete story.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats as sp_stats
import glob, os

rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
FIGDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"

def find_best_csv(target_mase, tol=0.01):
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

# Load baseline and MS46
bl_path = find_best_csv(1.2421)
ms_path = find_best_csv(1.1675)

bl_df = pd.read_csv(bl_path)
ms_df = pd.read_csv(ms_path)

mase_col = [c for c in bl_df.columns if 'MASE' in c and '0.5' in c]
if not mase_col:
    mase_col = [c for c in bl_df.columns if 'MASE' in c]
mase_col = mase_col[0]

bl_mase = bl_df[mase_col].values
ms_mase = ms_df[mase_col].values
improvement = (bl_mase - ms_mase) / bl_mase * 100

fig = plt.figure(figsize=(12, 4))

# Panel A: Per-config improvement distribution
ax1 = fig.add_subplot(131)
improvements_sorted = np.sort(improvement)
colors = ['#2ca02c' if x > 0 else '#d62728' for x in improvements_sorted]
ax1.bar(range(len(improvements_sorted)), improvements_sorted, color=colors, width=1, alpha=0.7)
ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.axhline(y=np.mean(improvement), color='#ff7f0e', linewidth=1.5, linestyle='--',
            label=f'Mean: {np.mean(improvement):+.1f}%')
ax1.axhline(y=np.median(improvement), color='#9467bd', linewidth=1.5, linestyle=':',
            label=f'Median: {np.median(improvement):+.1f}%')

win_rate = np.sum(improvement > 0) / len(improvement) * 100
ax1.text(0.95, 0.05, f'Win: {int(np.sum(improvement > 0))}/97\n({win_rate:.0f}%)',
         transform=ax1.transAxes, ha='right', va='bottom', fontsize=9, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_xlabel('Configuration (sorted)')
ax1.set_ylabel('MASE improvement (%)')
ax1.set_title('(a) Per-config improvement (n=97)')
ax1.legend(loc='upper left', fontsize=7)
ax1.grid(True, alpha=0.15, axis='y')

# Panel B: Horizon scaling
ax2 = fig.add_subplot(132)
horizons = ['short', 'medium', 'long']
horizon_labels = ['Short\n(n=55)', 'Medium\n(n=21)', 'Long\n(n=21)']
horizon_colors = ['#55A868', '#4C72B0', '#C44E52']

means = []
cis = []
win_rates = []
for h in horizons:
    mask = bl_df['term'] == h
    h_imp = improvement[mask.values]
    means.append(np.mean(h_imp))
    # Bootstrap CI
    boot = [np.mean(np.random.choice(h_imp, len(h_imp), replace=True)) for _ in range(5000)]
    cis.append((np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
    win_rates.append(np.sum(h_imp > 0) / len(h_imp) * 100)

x = np.arange(len(horizons))
bars = ax2.bar(x, means, color=horizon_colors, edgecolor='black', linewidth=0.5, width=0.6, alpha=0.8)
for i, (m, (lo, hi)) in enumerate(zip(means, cis)):
    ax2.errorbar(x[i], m, yerr=[[m-lo], [hi-m]], fmt='none', color='black', capsize=5, linewidth=1.5)
    ax2.text(x[i], hi + 0.5, f'{win_rates[i]:.0f}%\nwin', ha='center', fontsize=7, color='gray')

ax2.set_xticks(x)
ax2.set_xticklabels(horizon_labels)
ax2.axhline(y=0, color='black', linewidth=0.5, linestyle=':')
ax2.set_ylabel('Mean MASE improvement (%)')
ax2.set_title('(b) Improvement scales with horizon')
ax2.grid(True, alpha=0.15, axis='y')

# Panel C: Ablation summary (horizontal bars)
ax3 = fig.add_subplot(133)
BASELINE = 1.2421
ablation_data = [
    ('MS46 (Cheb d=4+d=6)', 1.1675),
    ('Cheb d=4 + 10% drop', 1.1802),
    ('L2-opt d=6', 1.1784),
    ('Duplicate input', 1.2342),
    ('Learned 4-tap', 1.2351),
    ('Learned 16-tap', 1.2775),
    ('Random hints', 1.2806),
    ('Zero hints', 1.3201),
]

names = [a[0] for a in ablation_data]
imps = [(BASELINE - a[1]) / BASELINE * 100 for a in ablation_data]
pvals = [8.5e-8, None, None, 0.495, 0.131, 0.108, None, None]  # from significance tests

y = np.arange(len(names))
bar_colors = ['#1f77b4' if imp > 1 else '#ff7f0e' if imp > 0 else '#d62728' for imp in imps]

bars = ax3.barh(y, imps, color=bar_colors, edgecolor='black', linewidth=0.5, height=0.6, alpha=0.8)
ax3.axvline(x=0, color='black', linewidth=0.8)

# Add significance markers
for i, (name, imp, p) in enumerate(zip(names, imps, pvals)):
    if p is not None and p < 0.001:
        marker = '***'
    elif p is not None and p < 0.05:
        marker = '*'
    elif p is not None:
        marker = 'ns'
    else:
        marker = ''
    offset = 0.3 if imp >= 0 else -0.3
    ha = 'left' if imp >= 0 else 'right'
    ax3.text(imp + offset, i, f'{imp:+.1f}% {marker}', ha=ha, va='center', fontsize=7)

ax3.set_yticks(y)
ax3.set_yticklabels(names, fontsize=8)
ax3.set_xlabel('MASE improvement (%)')
ax3.set_title('(c) Ablation: structure matters')
ax3.grid(True, alpha=0.15, axis='x')
ax3.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{FIGDIR}/main_result.pdf')
plt.savefig(f'{FIGDIR}/main_result.png')
print("Saved main_result.pdf/png")
