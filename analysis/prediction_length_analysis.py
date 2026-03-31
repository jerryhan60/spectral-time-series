#!/usr/bin/env python3
"""
Prediction length analysis: How does hint improvement scale with absolute
prediction length and prediction_length/context_length ratio?

Creates:
    - figures/pred_length_improvement.pdf: Improvement vs prediction length
    - figures/pred_ratio_improvement.pdf: Improvement vs pred/ctx ratio

Usage:
    python analysis/prediction_length_analysis.py
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
os.makedirs(FIG_DIR, exist_ok=True)

BL_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260219_121557.csv")
MS_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260223_163428.csv")


def main():
    bl = pd.read_csv(BL_CSV)
    ms = pd.read_csv(MS_CSV)

    mase_col = [c for c in bl.columns if 'MASE' in c and '0.5' in c]
    if not mase_col:
        mase_col = [c for c in bl.columns if 'MASE' in c]
    mase_col = mase_col[0]

    bl = bl.rename(columns={mase_col: 'bl_mase'})
    ms = ms.rename(columns={mase_col: 'ms_mase'})

    merged = bl[['dataset', 'term', 'frequency', 'prediction_length', 'bl_mase']].merge(
        ms[['dataset', 'term', 'ms_mase']], on=['dataset', 'term'], how='inner'
    )

    merged = merged.dropna(subset=['bl_mase', 'ms_mase', 'prediction_length'])
    merged = merged[(merged['bl_mase'] > 0) & (merged['ms_mase'] > 0)]
    merged['improvement'] = (merged['bl_mase'] - merged['ms_mase']) / merged['bl_mase'] * 100
    merged['log_pred_len'] = np.log10(merged['prediction_length'])

    print(f"Loaded {len(merged)} paired configs")
    print(f"Prediction length range: {merged['prediction_length'].min()} to {merged['prediction_length'].max()}")

    # === Figure 1: Improvement vs Prediction Length ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): Scatter + trend
    ax = axes[0]
    term_colors = {'short': '#55A868', 'medium': '#4C72B0', 'long': '#C44E52'}
    for term in ['short', 'medium', 'long']:
        sub = merged[merged['term'] == term]
        ax.scatter(sub['prediction_length'], sub['improvement'],
                  c=term_colors[term], label=term.capitalize(),
                  alpha=0.6, s=25, edgecolors='black', linewidths=0.3)

    # Log-linear trend
    from numpy.polynomial import polynomial as P
    log_pl = np.log(merged['prediction_length'].values)
    imp = merged['improvement'].values
    coeffs = np.polyfit(log_pl, imp, 1)
    x_fit = np.linspace(log_pl.min(), log_pl.max(), 100)
    ax.plot(np.exp(x_fit), np.polyval(coeffs, x_fit), 'k--', linewidth=1.5,
            label=f'Trend (slope={coeffs[0]:.2f})')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Prediction Length')
    ax.set_ylabel('MASE Improvement (%)')
    ax.set_title('(a) Improvement vs Prediction Length')
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (b): Binned improvement by prediction length
    ax = axes[1]
    bins = [0, 12, 24, 48, 96, 192, 1000]
    bin_labels = ['1-12', '13-24', '25-48', '49-96', '97-192', '193+']
    merged['pl_bin'] = pd.cut(merged['prediction_length'], bins=bins, labels=bin_labels, right=True)

    bin_stats = merged.groupby('pl_bin', observed=True).agg(
        mean_imp=('improvement', 'mean'),
        std_imp=('improvement', 'std'),
        count=('improvement', 'count'),
        wins=('improvement', lambda x: (x > 0).sum()),
    ).reset_index()

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(bin_stats)))
    x = np.arange(len(bin_stats))
    bars = ax.bar(x, bin_stats['mean_imp'], color=colors, edgecolor='black', linewidth=0.5,
           yerr=bin_stats['std_imp'] / np.sqrt(bin_stats['count']),
           capsize=3, ecolor='gray')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_stats['pl_bin'], fontsize=9)
    ax.set_xlabel('Prediction Length Bin')
    ax.set_ylabel('Mean Improvement (%)')
    ax.set_title('(b) Binned by Prediction Length')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add count labels
    for i, (_, row) in enumerate(bin_stats.iterrows()):
        ax.text(i, row['mean_imp'] + 1, f'n={int(row["count"])}',
               ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'pred_length_improvement.{ext}'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved pred_length_improvement.pdf/png")

    # === Frequency analysis ===
    fig, ax = plt.subplots(figsize=(7, 4))

    freq_order = ['5T', '10T', '15T', '30T', 'H', 'D', 'W', 'M', 'Q']
    freq_labels = ['5min', '10min', '15min', '30min', 'Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly']

    freq_stats = []
    for freq in freq_order:
        sub = merged[merged['frequency'] == freq]
        if len(sub) >= 2:
            freq_stats.append({
                'freq': freq,
                'mean': sub['improvement'].mean(),
                'se': sub['improvement'].std() / np.sqrt(len(sub)),
                'n': len(sub),
                'wins': (sub['improvement'] > 0).sum(),
            })

    fs_df = pd.DataFrame(freq_stats)
    x = np.arange(len(fs_df))
    colors = ['#55A868' if m > 0 else '#C44E52' for m in fs_df['mean']]
    ax.bar(x, fs_df['mean'], color=colors, edgecolor='black', linewidth=0.5,
           yerr=fs_df['se'], capsize=3, ecolor='gray')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    labels = [freq_labels[freq_order.index(f)] for f in fs_df['freq']]
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('Mean MASE Improvement (%)')
    ax.set_title('Improvement by Data Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (_, row) in enumerate(fs_df.iterrows()):
        ax.text(i, row['mean'] + 0.5 if row['mean'] > 0 else row['mean'] - 1.5,
               f"{int(row['wins'])}/{int(row['n'])}",
               ha='center', va='bottom' if row['mean'] > 0 else 'top', fontsize=7)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'freq_improvement.{ext}'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved freq_improvement.pdf/png")

    # Print summary statistics
    print("\n=== Prediction Length Correlation ===")
    corr = np.corrcoef(np.log(merged['prediction_length']), merged['improvement'])[0, 1]
    print(f"Pearson r (log pred_len vs improvement): {corr:.3f}")

    print("\n=== Per-Frequency Summary ===")
    print(f"{'Freq':<10} {'Mean Imp':>10} {'Wins/n':>10}")
    for _, row in fs_df.iterrows():
        print(f"{row['freq']:<10} {row['mean']:>+9.1f}% {int(row['wins'])}/{int(row['n']):>8}")


if __name__ == "__main__":
    main()
