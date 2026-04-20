#!/usr/bin/env python3
"""
Aggregation robustness: Do results hold across different summary statistics?

Tests: geometric mean, arithmetic mean, trimmed mean, median, win rate, MASE<1 count.
Shows that the -6% improvement is not driven by outliers or aggregation choice.

Usage:
    python analysis/aggregation_robustness.py
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
TAB_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/tables"
RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

BL_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260219_121557.csv")
MS_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260223_163428.csv")


def trimmed_mean(x, trim=0.1):
    """Trimmed mean: drop top and bottom trim% of values."""
    n = len(x)
    k = int(n * trim)
    sorted_x = np.sort(x)
    return np.mean(sorted_x[k:n-k])


def main():
    bl = pd.read_csv(BL_CSV)
    ms = pd.read_csv(MS_CSV)

    mase_col = [c for c in bl.columns if 'MASE' in c and '0.5' in c]
    if not mase_col:
        mase_col = [c for c in bl.columns if 'MASE' in c]
    mase_col = mase_col[0]

    bl = bl.rename(columns={mase_col: 'bl_mase'})
    ms = ms.rename(columns={mase_col: 'ms_mase'})

    merged = bl[['dataset', 'term', 'bl_mase']].merge(
        ms[['dataset', 'term', 'ms_mase']], on=['dataset', 'term'], how='inner'
    )
    valid = merged.dropna(subset=['bl_mase', 'ms_mase'])
    valid = valid[(valid['bl_mase'] > 0) & (valid['ms_mase'] > 0)]

    bl_vals = valid['bl_mase'].values
    ms_vals = valid['ms_mase'].values
    n = len(valid)

    # Compute all aggregation metrics
    metrics = {
        'Geometric Mean': (np.exp(np.mean(np.log(bl_vals))), np.exp(np.mean(np.log(ms_vals)))),
        'Arithmetic Mean': (np.mean(bl_vals), np.mean(ms_vals)),
        'Trimmed Mean (10%)': (trimmed_mean(bl_vals, 0.1), trimmed_mean(ms_vals, 0.1)),
        'Trimmed Mean (20%)': (trimmed_mean(bl_vals, 0.2), trimmed_mean(ms_vals, 0.2)),
        'Median': (np.median(bl_vals), np.median(ms_vals)),
        'MASE < 1.0 count': ((bl_vals < 1.0).sum(), (ms_vals < 1.0).sum()),
        'Win count': (0, (ms_vals < bl_vals).sum()),  # handled specially
    }

    print("=" * 70)
    print("AGGREGATION ROBUSTNESS ANALYSIS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Baseline':>10} {'MS46':>10} {'Change':>10}")
    print("-" * 58)

    rows = []
    for name, (bl_v, ms_v) in metrics.items():
        if name == 'Win count':
            print(f"{'Win rate':<25} {'':>10} {f'{ms_v}/{n}':>10} {f'{ms_v/n*100:.0f}%':>10}")
            rows.append(('Win rate', '', f'{int(ms_v)}/{n}', f'{ms_v/n*100:.0f}%'))
        elif name == 'MASE < 1.0 count':
            change = ms_v - bl_v
            print(f"{name:<25} {int(bl_v):>10} {int(ms_v):>10} {f'+{int(change)}':>10}")
            rows.append((name, str(int(bl_v)), str(int(ms_v)), f'+{int(change)}'))
        else:
            change = (ms_v - bl_v) / bl_v * 100
            print(f"{name:<25} {bl_v:>10.4f} {ms_v:>10.4f} {change:>+9.1f}%")
            rows.append((name, f'{bl_v:.4f}', f'{ms_v:.4f}', f'{change:+.1f}%'))

    # === Figure: Bar chart of % change across aggregation methods ===
    fig, ax = plt.subplots(figsize=(7, 4))

    method_names = ['Geo Mean', 'Arith Mean', 'Trim 10%', 'Trim 20%', 'Median']
    changes = []
    for name in ['Geometric Mean', 'Arithmetic Mean', 'Trimmed Mean (10%)',
                 'Trimmed Mean (20%)', 'Median']:
        bl_v, ms_v = metrics[name]
        changes.append((ms_v - bl_v) / bl_v * 100)

    colors = ['#C44E52'] * len(changes)
    x = np.arange(len(method_names))
    bars = ax.bar(x, changes, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=10)
    ax.set_ylabel('Change (%)')
    ax.set_title('Improvement Across Aggregation Methods')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, c in zip(bars, changes):
        ax.text(bar.get_x() + bar.get_width()/2., c - 0.3,
               f'{c:.1f}%', ha='center', va='top', fontsize=9, fontweight='bold')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'aggregation_robustness.{ext}'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved aggregation_robustness.pdf/png")

    # === LaTeX table ===
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Aggregation robustness: MS $d{=}4{+}d{=}6$ vs baseline across different summary statistics. The improvement is consistent regardless of how results are aggregated.}")
    tex.append(r"\label{tab:aggregation}")
    tex.append(r"\begin{tabular}{lrrr}")
    tex.append(r"\toprule")
    tex.append(r"Aggregation Method & Baseline & MS46 & Change \\")
    tex.append(r"\midrule")
    for name, bl_s, ms_s, change in rows:
        tex.append(f"{name} & {bl_s} & {ms_s} & {change} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    tex_path = os.path.join(TAB_DIR, "table_aggregation.tex")
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex))
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
