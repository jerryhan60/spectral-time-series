#!/usr/bin/env python3
"""
Statistical significance analysis for polynomial hint preconditioning.

1. Paired bootstrap test (ms46 vs baseline)
2. Per-dataset scatter plot
3. Win/loss analysis by dataset characteristics

Usage:
    python analysis/statistical_analysis.py
"""

import numpy as np
import pandas as pd
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Known CSV paths (identified from reports)
BL_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260219_121557.csv")
MS_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260223_163428.csv")


def load_mase(csv_path):
    df = pd.read_csv(csv_path)
    mase_col = [c for c in df.columns if 'MASE' in c and '0.5' in c]
    if not mase_col:
        mase_col = [c for c in df.columns if 'MASE' in c]
    return df, mase_col[0]


def paired_bootstrap(a, b, n_boot=10000, seed=42):
    """Bootstrap test: P(geomean(a) < geomean(b))."""
    rng = np.random.RandomState(seed)
    log_diff = np.log(a) - np.log(b)
    n = len(log_diff)
    boot = np.array([np.mean(log_diff[rng.choice(n, n, replace=True)]) for _ in range(n_boot)])
    return {
        'ratio': np.exp(np.mean(log_diff)),
        'ci_low': np.exp(np.percentile(boot, 2.5)),
        'ci_high': np.exp(np.percentile(boot, 97.5)),
        'p_a_better': np.mean(boot < 0),
        'p_b_better': np.mean(boot > 0),
    }


def main():
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)

    # Load both CSVs
    bl_df, bl_mase_col = load_mase(BL_CSV)
    ms_df, ms_mase_col = load_mase(MS_CSV)

    # Merge on config_name to get paired values
    bl_df = bl_df.rename(columns={bl_mase_col: 'baseline_mase'})
    ms_df = ms_df.rename(columns={ms_mase_col: 'ms46_mase'})

    merged = bl_df[['config_name', 'baseline_mase']].merge(
        ms_df[['config_name', 'ms46_mase']], on='config_name', how='inner'
    )

    # Add term/frequency info
    if 'term' in bl_df.columns:
        merged = merged.merge(bl_df[['config_name', 'term', 'frequency', 'dataset']].drop_duplicates(),
                              on='config_name', how='left')

    # Filter valid
    valid = merged.dropna(subset=['baseline_mase', 'ms46_mase'])
    valid = valid[(valid['baseline_mase'] > 0) & (valid['ms46_mase'] > 0)]

    print(f"\nPaired configs: {len(valid)}")
    print(f"Baseline geo-mean MASE:  {np.exp(np.mean(np.log(valid['baseline_mase']))):.4f}")
    print(f"MS46 geo-mean MASE:      {np.exp(np.mean(np.log(valid['ms46_mase']))):.4f}")

    # Win/loss
    wins = (valid['ms46_mase'] < valid['baseline_mase']).sum()
    losses = (valid['ms46_mase'] > valid['baseline_mase']).sum()
    ties = (valid['ms46_mase'] == valid['baseline_mase']).sum()
    print(f"Win/Loss/Tie: {wins}/{losses}/{ties} ({wins/len(valid)*100:.0f}% win rate)")

    # Bootstrap test
    print("\n--- Paired Bootstrap Test (10K resamples) ---")
    result = paired_bootstrap(valid['ms46_mase'].values, valid['baseline_mase'].values)
    print(f"Ratio (ms46/baseline): {result['ratio']:.4f}")
    print(f"95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    print(f"P(ms46 better): {result['p_a_better']:.4f}")
    sig = "***" if result['p_a_better'] > 0.999 else "**" if result['p_a_better'] > 0.99 else "*" if result['p_a_better'] > 0.95 else "n.s."
    print(f"Significance: {sig}")

    # Wilcoxon signed-rank test
    log_diffs = np.log(valid['ms46_mase'].values) - np.log(valid['baseline_mase'].values)
    try:
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(log_diffs)
        print(f"\n--- Wilcoxon Signed-Rank Test ---")
        print(f"Statistic: {stat:.1f}, p-value: {pval:.6f}")
        sig_w = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        print(f"Significance: {sig_w}")
    except ImportError:
        pval = 1 - result['p_a_better']  # Use bootstrap p-value as fallback
        print("\n(scipy not available, skipping Wilcoxon test)")

    # Per-horizon analysis
    if 'term' in valid.columns:
        print("\n--- Per-Horizon Bootstrap ---")
        for term in ['short', 'medium', 'long']:
            subset = valid[valid['term'] == term]
            if len(subset) > 3:
                r = paired_bootstrap(subset['ms46_mase'].values, subset['baseline_mase'].values)
                wins_h = (subset['ms46_mase'] < subset['baseline_mase']).sum()
                print(f"  {term:>8}: n={len(subset):>2}, ratio={r['ratio']:.4f}, "
                      f"CI=[{r['ci_low']:.4f},{r['ci_high']:.4f}], "
                      f"P(better)={r['p_a_better']:.3f}, wins={wins_h}/{len(subset)}")

    # ============== Figures ==============
    if HAS_MPL:
        # Scatter plot
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = []
        if 'term' in valid.columns:
            term_colors = {'short': '#55A868', 'medium': '#4C72B0', 'long': '#C44E52'}
            colors = [term_colors.get(t, 'gray') for t in valid['term']]
        else:
            colors = ['#4C72B0'] * len(valid)

        ax.scatter(valid['baseline_mase'], valid['ms46_mase'], c=colors, alpha=0.6, s=30, edgecolors='black', linewidths=0.3)

        lims = [0, max(valid['baseline_mase'].max(), valid['ms46_mase'].max()) * 1.05]
        # Cap at reasonable range for readability
        lims[1] = min(lims[1], 10)
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Baseline MASE')
        ax.set_ylabel('MS d=4+d=6 MASE')
        ax.set_title(f'Per-Config MASE: Hint vs Baseline\n(Win: {wins}/{len(valid)}, p < {pval:.4f})')
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend for term colors
        if 'term' in valid.columns:
            for term, color in term_colors.items():
                ax.scatter([], [], c=color, label=term.capitalize(), s=30, edgecolors='black', linewidths=0.3)
            ax.legend(frameon=False, loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "scatter_mase.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, "scatter_mase.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nWrote {FIG_DIR}/scatter_mase.pdf")

        # Improvement distribution
        fig, ax = plt.subplots(figsize=(5, 3.5))
        improvements = (valid['baseline_mase'] - valid['ms46_mase']) / valid['baseline_mase'] * 100
        ax.hist(improvements, bins=30, color='#4C72B0', edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.axvline(x=0, color='red', linewidth=1, linestyle='--')
        ax.axvline(x=improvements.median(), color='green', linewidth=1.5, label=f'Median: {improvements.median():.1f}%')
        ax.set_xlabel('Improvement over Baseline (%)')
        ax.set_ylabel('Number of Configs')
        ax.set_title('Distribution of Per-Config Improvement')
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "improvement_dist.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, "improvement_dist.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Wrote {FIG_DIR}/improvement_dist.pdf")

    # Top wins and losses
    valid = valid.copy()
    valid['improvement_pct'] = (valid['baseline_mase'] - valid['ms46_mase']) / valid['baseline_mase'] * 100

    print("\n--- Top 10 Improvements ---")
    top = valid.nlargest(10, 'improvement_pct')
    for _, row in top.iterrows():
        name = row.get('dataset', row['config_name'])
        term = row.get('term', '?')
        print(f"  {name:40s} {term:>8} BL={row['baseline_mase']:.4f} → {row['ms46_mase']:.4f} ({row['improvement_pct']:+.1f}%)")

    print("\n--- Top 10 Regressions ---")
    bot = valid.nsmallest(10, 'improvement_pct')
    for _, row in bot.iterrows():
        name = row.get('dataset', row['config_name'])
        term = row.get('term', '?')
        print(f"  {name:40s} {term:>8} BL={row['baseline_mase']:.4f} → {row['ms46_mase']:.4f} ({row['improvement_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
