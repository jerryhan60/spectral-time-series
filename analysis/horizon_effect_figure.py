"""
Prediction horizon effect figure: HD10 benefit increases with horizon length.
Uses 5-seed pooled data.

Generates:
  - analysis/figures/horizon_effect.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gmean, binomtest

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"

BL = {0: '20260308_000815', 1: '20260308_002546', 2: '20260308_030127', 7: '20260308_190028', 42: '20260309_011307'}
HD = {0: '20260308_005524', 1: '20260308_011431', 2: '20260308_034842', 7: '20260308_194312', 42: '20260309_033157'}

all_merged = []
for seed in [0, 1, 2, 7, 42]:
    bl = pd.read_csv(f'{RESULTS_DIR}/gifteval_results_epoch_99-step_10000_{BL[seed]}.csv')
    hd = pd.read_csv(f'{RESULTS_DIR}/gifteval_results_epoch_99-step_10000_{HD[seed]}.csv')
    bl['config'] = bl['dataset'] + '/' + bl['term']
    hd['config'] = hd['dataset'] + '/' + hd['term']
    m = bl[['config','dataset','term','frequency','prediction_length','MASE']].merge(
        hd[['config','MASE']], on='config', suffixes=('_bl','_hd'))
    m['seed'] = seed
    all_merged.append(m)
df = pd.concat(all_merged)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 13, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): By term (short/medium/long)
terms = ['short', 'medium', 'long']
improvements = []
win_rates = []
p_vals = []
ns = []
for term in terms:
    sub = df[df['term'] == term]
    gm_bl = gmean(sub['MASE_bl'])
    gm_hd = gmean(sub['MASE_hd'])
    imp = (1 - gm_hd / gm_bl) * 100
    wins = int((sub['MASE_hd'] < sub['MASE_bl']).sum())
    n = len(sub)
    p = binomtest(wins, n, 0.5, alternative='greater').pvalue
    improvements.append(imp)
    win_rates.append(wins / n * 100)
    p_vals.append(p)
    ns.append(n)

x = np.arange(len(terms))
colors = ['#2E7D32'] * 3
bars = ax1.bar(x, improvements, color=colors, edgecolor='white', width=0.5, alpha=0.85)

for i, (v, n, p) in enumerate(zip(improvements, ns, p_vals)):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax1.text(i, v + 0.15, f'+{v:.1f}%{sig}\n(n={n})', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(['Short\n(1x base)', 'Medium\n(10x base)', 'Long\n(15x base)'])
ax1.set_ylabel('MASE Improvement (%)')
ax1.set_title('(a) Improvement by Prediction Horizon')
ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.grid(True, alpha=0.2, axis='y')
ax1.set_ylim(0, 7)

# Panel (b): Win rate by term
bars2 = ax2.bar(x, win_rates, color=['#ff7f0e'] * 3, edgecolor='white', width=0.5, alpha=0.85)
ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Random (50%)')
for i, (wr, n) in enumerate(zip(win_rates, ns)):
    ax2.text(i, wr + 0.5, f'{wr:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(['Short', 'Medium', 'Long'])
ax2.set_ylabel('Win Rate vs Baseline (%)')
ax2.set_title('(b) Win Rate by Prediction Horizon')
ax2.legend(frameon=False, loc='lower right')
ax2.grid(True, alpha=0.2, axis='y')
ax2.set_ylim(40, 80)

plt.suptitle('Longer Prediction Horizons Benefit More from Polynomial Hints\n(HD10 vs BL, 5 seeds pooled)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIG_DIR, f'horizon_effect.{ext}'), bbox_inches='tight')
plt.close(fig)
print("Saved horizon_effect.pdf/png")
