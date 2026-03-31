"""
Per-horizon analysis figure for publication.

Shows how hint preconditioning benefit scales with forecast horizon:
short → medium → long.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob, os

rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
FIGDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"

def find_best_csv(target_mase):
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
            if diff < best_diff:
                best_diff = diff
                best_path = csv_path
    return best_path

bl_path = find_best_csv(1.2421)
ms_path = find_best_csv(1.1675)

bl_df = pd.read_csv(bl_path)
ms_df = pd.read_csv(ms_path)

mase_col = [c for c in bl_df.columns if 'MASE' in c and '0.5' in c]
if not mase_col:
    mase_col = [c for c in bl_df.columns if 'MASE' in c]
mase_col = mase_col[0]

# Compute per-config improvement
paired = pd.DataFrame({
    'dataset': bl_df['dataset'],
    'term': bl_df.get('term', ''),
    'frequency': bl_df.get('frequency', ''),
    'bl_mase': bl_df[mase_col].values,
    'ms_mase': ms_df[mase_col].values,
})
paired['improvement_pct'] = (paired['bl_mase'] - paired['ms_mase']) / paired['bl_mase'] * 100

fig, axes = plt.subplots(1, 3, figsize=(11, 4))

# Panel A: Per-horizon boxplot
ax = axes[0]
horizons = ['short', 'medium', 'long']
horizon_data = []
for h in horizons:
    mask = paired['term'] == h
    horizon_data.append(paired.loc[mask, 'improvement_pct'].values)

bp = ax.boxplot(horizon_data, labels=['Short', 'Medium', 'Long'],
                patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2))

colors = ['#55A868', '#4C72B0', '#C44E52']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(y=0, color='black', linewidth=0.5, linestyle=':')
ax.set_ylabel('MASE improvement (%)')
ax.set_title('(a) Improvement by horizon')
ax.grid(True, alpha=0.2, axis='y')

# Annotate means and win rates
for i, (h, data) in enumerate(zip(horizons, horizon_data)):
    mean_imp = np.mean(data)
    win_rate = np.sum(data > 0) / len(data) * 100
    ax.text(i + 1, ax.get_ylim()[1] * 0.9,
            f'Mean: {mean_imp:+.1f}%\nWin: {win_rate:.0f}%',
            ha='center', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

# Panel B: Per-horizon scatter
ax = axes[1]
for h, color, marker in zip(horizons, colors, ['o', 's', 'D']):
    mask = paired['term'] == h
    subset = paired[mask]
    ax.scatter(subset['bl_mase'], subset['ms_mase'], c=color, s=25,
              alpha=0.6, marker=marker, edgecolors='none', label=h.capitalize())

# Diagonal
lims = [0, max(paired['bl_mase'].max(), paired['ms_mase'].max()) * 1.05]
ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Baseline MASE')
ax.set_ylabel('MS46 MASE')
ax.set_title('(b) Per-config paired comparison')
ax.legend(loc='upper left', framealpha=0.8)
ax.set_xlim(lims)
ax.set_ylim(lims)

# Panel C: Win rate by frequency and horizon
ax = axes[2]
freq_horizon = paired.groupby(['frequency', 'term']).agg(
    n=('improvement_pct', 'count'),
    wins=('improvement_pct', lambda x: (x > 0).sum()),
    mean_imp=('improvement_pct', 'mean'),
).reset_index()
freq_horizon['win_rate'] = freq_horizon['wins'] / freq_horizon['n'] * 100

# Pivot for heatmap
from scipy.stats import pearsonr

# Simple version: plot improvement vs horizon for each frequency group
freq_groups = {
    'Sub-hourly': ['10S', '5T', '10T', '15T'],
    'Hourly': ['H'],
    'Daily': ['D'],
    'Weekly+': ['W-SUN', 'W-TUE', 'W-THU', 'W-FRI', 'W-WED'],
    'Monthly+': ['M', 'Q-DEC', 'A-DEC', 'Y'],
}

x_pos = 0
x_ticks = []
x_labels = []
bar_width = 0.25

for freq_name, freq_list in freq_groups.items():
    mask = paired['frequency'].isin(freq_list)
    if mask.sum() == 0:
        continue
    subset = paired[mask]

    for j, h in enumerate(horizons):
        h_mask = subset['term'] == h
        if h_mask.sum() == 0:
            continue
        mean_imp = subset.loc[h_mask, 'improvement_pct'].mean()
        color = colors[j]
        ax.bar(x_pos + j * bar_width, mean_imp, bar_width * 0.9,
               color=color, alpha=0.7, edgecolor='black', linewidth=0.3)

    x_ticks.append(x_pos + bar_width)
    x_labels.append(freq_name)
    x_pos += 1.2

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontsize=8, rotation=15)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle=':')
ax.set_ylabel('Mean MASE improvement (%)')
ax.set_title('(c) By frequency group × horizon')

# Legend for horizons
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, alpha=0.7, label=h.capitalize())
                   for c, h in zip(colors, horizons)]
ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.8)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/horizon_analysis.pdf')
plt.savefig(f'{FIGDIR}/horizon_analysis.png')
print("Saved horizon_analysis.pdf/png")

# Print statistics
print("\nPer-horizon summary:")
for h, data in zip(horizons, horizon_data):
    n = len(data)
    wins = np.sum(data > 0)
    print(f"  {h:8s}: n={n}, mean={np.mean(data):+.2f}%, median={np.median(data):+.2f}%, "
          f"win={wins}/{n} ({wins/n*100:.0f}%)")
