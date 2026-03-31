#!/usr/bin/env python3
"""
Per-config MASE heatmap: Where do hints help vs hurt?

Creates a heatmap organized by domain x horizon showing % improvement
for each of the 97 GIFT-Eval configurations.

Usage:
    python analysis/heatmap_analysis.py
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
os.makedirs(FIG_DIR, exist_ok=True)

BL_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260219_121557.csv")
MS_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260223_163428.csv")


def categorize_domain(dataset):
    ds = dataset.lower()
    if 'solar' in ds:
        return 'Solar'
    if 'elec' in ds:
        return 'Electricity'
    if any(x in ds for x in ['traffic', 'taxi', 'uber', 'rideshare', 'loop']):
        return 'Transport'
    if 'pedestrian' in ds:
        return 'Pedestrian'
    if any(x in ds for x in ['bitbrains', 'cloud']):
        return 'IT/Cloud'
    if any(x in ds for x in ['weather', 'temp']):
        return 'Weather'
    if 'ett' in ds:
        return 'ETT'
    if any(x in ds for x in ['m4', 'm1', 'm3']):
        return 'Competition'
    if 'tourism' in ds:
        return 'Tourism'
    if 'hospital' in ds:
        return 'Healthcare'
    if 'covid' in ds:
        return 'COVID'
    return 'Other'


def main():
    bl = pd.read_csv(BL_CSV)
    ms = pd.read_csv(MS_CSV)

    mase_col = [c for c in bl.columns if 'MASE' in c and '0.5' in c]
    if not mase_col:
        mase_col = [c for c in bl.columns if 'MASE' in c]
    mase_col = mase_col[0]

    bl = bl.rename(columns={mase_col: 'bl_mase'})
    ms = ms.rename(columns={mase_col: 'ms_mase'})

    merged = bl[['dataset', 'term', 'frequency', 'bl_mase']].merge(
        ms[['dataset', 'term', 'ms_mase']], on=['dataset', 'term'], how='inner'
    )

    merged = merged.dropna(subset=['bl_mase', 'ms_mase'])
    merged = merged[(merged['bl_mase'] > 0) & (merged['ms_mase'] > 0)]

    # Compute improvement
    merged['improvement'] = (merged['bl_mase'] - merged['ms_mase']) / merged['bl_mase'] * 100
    merged['domain'] = merged['dataset'].apply(categorize_domain)

    # Sort domains by mean improvement
    domain_order = merged.groupby('domain')['improvement'].mean().sort_values(ascending=False).index.tolist()
    horizon_order = ['short', 'medium', 'long']

    # Create pivot-like structure
    # For each domain-horizon pair, compute mean improvement and count
    cells = []
    for domain in domain_order:
        for horizon in horizon_order:
            subset = merged[(merged['domain'] == domain) & (merged['term'] == horizon)]
            if len(subset) > 0:
                cells.append({
                    'domain': domain,
                    'horizon': horizon,
                    'improvement': subset['improvement'].mean(),
                    'n': len(subset),
                    'wins': (subset['improvement'] > 0).sum(),
                })

    cells_df = pd.DataFrame(cells)

    # Pivot table
    pivot = cells_df.pivot(index='domain', columns='horizon', values='improvement')
    pivot = pivot.reindex(index=domain_order, columns=horizon_order)
    counts = cells_df.pivot(index='domain', columns='horizon', values='n')
    counts = counts.reindex(index=domain_order, columns=horizon_order)
    wins = cells_df.pivot(index='domain', columns='horizon', values='wins')
    wins = wins.reindex(index=domain_order, columns=horizon_order)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(7, 8))

    # Diverging colormap centered at 0
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(pivot.values, cmap='RdYlGn', norm=norm, aspect='auto')

    # Add text annotations
    for i in range(len(domain_order)):
        for j in range(len(horizon_order)):
            val = pivot.iloc[i, j]
            n = counts.iloc[i, j]
            w = wins.iloc[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:+.1f}%\n({int(w)}/{int(n)})',
                       ha='center', va='center', fontsize=7,
                       color=text_color, fontweight='bold' if abs(val) > 5 else 'normal')

    ax.set_xticks(range(len(horizon_order)))
    ax.set_xticklabels([h.capitalize() for h in horizon_order], fontsize=10)
    ax.set_yticks(range(len(domain_order)))
    ax.set_yticklabels(domain_order, fontsize=9)
    ax.set_xlabel('Forecast Horizon', fontsize=11)
    ax.set_ylabel('Domain', fontsize=11)
    ax.set_title('MASE Improvement (%): MS $d{=}4{+}d{=}6$ vs Baseline\n(wins/total per cell)', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Improvement (%)', fontsize=10)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'domain_horizon_heatmap.{ext}'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved domain_horizon_heatmap.pdf/png")

    # Also create a compact version: per-dataset ranked bar chart
    fig, ax = plt.subplots(figsize=(6, 12))

    # Sort all configs by improvement
    merged_sorted = merged.sort_values('improvement', ascending=True)

    colors = []
    for _, row in merged_sorted.iterrows():
        if row['improvement'] > 0:
            colors.append('#55A868')
        else:
            colors.append('#C44E52')

    y_pos = np.arange(len(merged_sorted))
    labels = [f"{row['dataset']}/{row['term']}" for _, row in merged_sorted.iterrows()]

    ax.barh(y_pos, merged_sorted['improvement'], color=colors, height=0.8, edgecolor='none')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_yticks(y_pos[::3])
    ax.set_yticklabels([labels[i] for i in range(0, len(labels), 3)], fontsize=5)
    ax.set_xlabel('Improvement (%)')
    ax.set_title('Per-Config MASE Change (all 97 configs)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add summary stats
    n_better = (merged_sorted['improvement'] > 0).sum()
    n_worse = (merged_sorted['improvement'] < 0).sum()
    ax.text(0.02, 0.98, f'Better: {n_better}\nWorse: {n_worse}',
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'per_config_waterfall.{ext}'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved per_config_waterfall.pdf/png")

    # Print domain x horizon summary table
    print("\n=== Domain x Horizon Improvement Summary ===")
    print(f"{'Domain':<15} {'Short':>8} {'Medium':>8} {'Long':>8} {'Overall':>8}")
    print("-" * 50)
    for domain in domain_order:
        sub = merged[merged['domain'] == domain]
        parts = []
        for h in horizon_order:
            hs = sub[sub['term'] == h]
            if len(hs) > 0:
                parts.append(f"{hs['improvement'].mean():+.1f}%")
            else:
                parts.append("---")
        overall = f"{sub['improvement'].mean():+.1f}%"
        print(f"{domain:<15} {parts[0]:>8} {parts[1]:>8} {parts[2]:>8} {overall:>8}")


if __name__ == "__main__":
    main()
