"""
Paired per-config comparison between baseline and MS46.

Creates:
1. Complete paired table (97 configs)
2. Win/loss analysis by domain
3. Cumulative distribution function of improvements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
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
FIGDIR = Path("/scratch/gpfs/EHAZAN/jh1161/analysis/figures")
TABLEDIR = Path("/scratch/gpfs/EHAZAN/jh1161/analysis/tables")
FIGDIR.mkdir(exist_ok=True)
TABLEDIR.mkdir(exist_ok=True)

def find_best_csv(target_mase):
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
            if diff < best_diff:
                best_diff = diff
                best_path = csv_path
    return best_path

# Load baseline and ms46
bl_path = find_best_csv(1.2421)
ms_path = find_best_csv(1.1675)

if bl_path is None or ms_path is None:
    print("ERROR: Could not find result CSVs")
    exit(1)

bl_df = pd.read_csv(bl_path)
ms_df = pd.read_csv(ms_path)

# Get MASE column
mase_col = [c for c in bl_df.columns if 'MASE' in c and '0.5' in c]
if not mase_col:
    mase_col = [c for c in bl_df.columns if 'MASE' in c]
mase_col = mase_col[0]

# Create paired comparison
paired = pd.DataFrame({
    'dataset': bl_df['dataset'] if 'dataset' in bl_df.columns else bl_df.iloc[:, 0],
    'term': bl_df.get('term', ''),
    'frequency': bl_df.get('frequency', ''),
    'config': bl_df.get('config_name', bl_df.get('dataset', '')),
    'bl_mase': bl_df[mase_col].values,
    'ms_mase': ms_df[mase_col].values,
})

paired['improvement_pct'] = (paired['bl_mase'] - paired['ms_mase']) / paired['bl_mase'] * 100
paired['log_ratio'] = np.log(paired['ms_mase']) - np.log(paired['bl_mase'])
paired['winner'] = np.where(paired['ms_mase'] < paired['bl_mase'], 'MS46', 'Baseline')

# Sort by improvement
paired_sorted = paired.sort_values('improvement_pct', ascending=False)

print(f"=== Paired Comparison: MS46 vs Baseline ===")
print(f"Configs: {len(paired)}")
print(f"MS46 wins: {(paired['winner'] == 'MS46').sum()}")
print(f"BL wins: {(paired['winner'] == 'Baseline').sum()}")
print(f"\nTop 10 improvements:")
for _, row in paired_sorted.head(10).iterrows():
    print(f"  {row['dataset']}/{row['frequency']}/{row['term']}: {row['improvement_pct']:+.1f}%")
print(f"\nTop 10 regressions:")
for _, row in paired_sorted.tail(10).iterrows():
    print(f"  {row['dataset']}/{row['frequency']}/{row['term']}: {row['improvement_pct']:+.1f}%")

# --- Figure: CDF of improvements ---
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Panel A: CDF
ax = axes[0]
improvements = np.sort(paired['improvement_pct'].values)
cdf = np.arange(1, len(improvements) + 1) / len(improvements)
ax.plot(improvements, cdf, 'b-', linewidth=2)
ax.axvline(x=0, color='black', linewidth=0.5, linestyle=':')
ax.axhline(y=0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

# Mark key percentiles
median_imp = np.median(improvements)
mean_imp = np.mean(improvements)
ax.axvline(x=median_imp, color='#ff7f0e', linewidth=1.5, linestyle='--', label=f'Median: {median_imp:.1f}%')
ax.axvline(x=mean_imp, color='#2ca02c', linewidth=1.5, linestyle='--', label=f'Mean: {mean_imp:.1f}%')

# Shade improvement region
ax.fill_between(improvements[improvements > 0], 0,
                cdf[improvements > 0], alpha=0.1, color='green')
ax.fill_between(improvements[improvements < 0], 0,
                cdf[improvements < 0], alpha=0.1, color='red')

# Mark win rate
win_rate = (paired['winner'] == 'MS46').sum() / len(paired) * 100
ax.text(0.95, 0.05, f'Win rate: {win_rate:.0f}%', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('MASE improvement (%)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('(a) CDF of per-config improvements')
ax.legend(loc='upper left', framealpha=0.8, fontsize=8)
ax.grid(True, alpha=0.2)

# Panel B: Domain-level win rate
ax = axes[1]
# Group by domain (first part of dataset name)
paired['domain'] = paired['dataset'].apply(lambda x: str(x).split('_')[0] if '_' in str(x) else str(x)[:10])

domain_stats = paired.groupby('domain').agg(
    n=('improvement_pct', 'count'),
    mean_imp=('improvement_pct', 'mean'),
    wins=('winner', lambda x: (x == 'MS46').sum()),
).reset_index()
domain_stats['win_rate'] = domain_stats['wins'] / domain_stats['n'] * 100
domain_stats = domain_stats.sort_values('mean_imp', ascending=False)

# Only show domains with 2+ configs
domain_stats = domain_stats[domain_stats['n'] >= 2]

y_pos = np.arange(len(domain_stats))
colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in domain_stats['mean_imp']]
ax.barh(y_pos, domain_stats['mean_imp'], color=colors, edgecolor='black', linewidth=0.5, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(domain_stats['domain'].values, fontsize=7)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel('Mean MASE improvement (%)')
ax.set_title('(b) Domain-level improvement')
ax.grid(True, alpha=0.2, axis='x')

# Add win rate labels
for i, (_, row) in enumerate(domain_stats.iterrows()):
    ax.text(max(0, row['mean_imp']) + 0.5, i, f'{int(row["wins"])}/{int(row["n"])}',
            va='center', fontsize=6, color='gray')

plt.tight_layout()
plt.savefig(FIGDIR / 'paired_comparison.pdf')
plt.savefig(FIGDIR / 'paired_comparison.png')
print(f"\nSaved paired_comparison.pdf/png")

# Save paired comparison CSV
paired_sorted.to_csv('/scratch/gpfs/EHAZAN/jh1161/analysis/paired_comparison.csv', index=False)
print("Saved paired_comparison.csv")

# Generate LaTeX table for top wins/losses
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Largest per-config MASE improvements and regressions. MS46 ($d{=}4{+}d{=}6$) vs.\ baseline at 10K steps.}")
lines.append(r"\label{tab:wins_losses}")
lines.append(r"\begin{tabular}{lllrr}")
lines.append(r"\toprule")
lines.append(r"Dataset & Freq & Horizon & $\Delta$ MASE & Outcome \\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{l}{\textit{Top 5 improvements}} \\")
for _, row in paired_sorted.head(5).iterrows():
    lines.append(f"{str(row['dataset'])[:15]} & {row['frequency']} & {row['term']} & {row['improvement_pct']:+.1f}\\% & MS46 \\\\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{l}{\textit{Top 5 regressions}} \\")
for _, row in paired_sorted.tail(5).iloc[::-1].iterrows():
    lines.append(f"{str(row['dataset'])[:15]} & {row['frequency']} & {row['term']} & {row['improvement_pct']:+.1f}\\% & Baseline \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
with open(TABLEDIR / 'table_wins_losses.tex', 'w') as f:
    f.write('\n'.join(lines))
print("Saved table_wins_losses.tex")
