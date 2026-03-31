"""
Frequency-dependent improvement analysis.

Shows that hint improvement correlates strongly with sampling frequency,
consistent with the spectral gap theory: higher sampling rate = more
high-frequency content above the patch Nyquist frequency.
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
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
FIGDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"


def find_csv_by_mase(target_mase, tol=0.01):
    all_csvs = glob.glob(os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_*.csv"))
    best_path, best_diff = None, 999
    for csv_path in all_csvs:
        df = pd.read_csv(csv_path)
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


# Sampling period in seconds for each frequency
FREQ_PERIOD = {
    '10S': 10,
    '5T': 300,
    '10T': 600,
    '15T': 900,
    'H': 3600,
    'D': 86400,
    'W': 604800,
    'M': 2592000,
    'Q': 7776000,
}


def main():
    # Load baseline and MS46
    bl_df = pd.read_csv(os.path.join(RESULTS_DIR, 'all_results_epoch_99-step_10000.csv'))
    ms_path = find_csv_by_mase(1.1675)
    ms_df = pd.read_csv(ms_path)
    mase_col = [c for c in ms_df.columns if 'MASE' in c][0]

    bl_mase = bl_df['MASE'].values
    ms_mase = ms_df[mase_col].values
    improvement = (bl_mase - ms_mase) / bl_mase * 100

    # Parse configs
    configs = bl_df['dataset_config'].values
    freqs = []
    horizons = []
    for c in configs:
        parts = c.split('/')
        freq = None
        for p in parts[1:]:
            if p in FREQ_PERIOD:
                freq = p
                break
        if freq is None:
            freq = 'other'
        freqs.append(freq)
        horizons.append(parts[-1] if parts[-1] in ['short', 'medium', 'long'] else 'short')

    freqs = np.array(freqs)
    horizons = np.array(horizons)

    # --- Figure: 2-panel frequency analysis ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Mean improvement by frequency (ordered by sampling period)
    freq_order = ['10S', '5T', '10T', '15T', 'H', 'D', 'W', 'M']
    freq_labels = ['10s', '5min', '10min', '15min', '1h', '1d', '1w', '1mo']
    means = []
    stds = []
    counts = []
    for freq in freq_order:
        mask = freqs == freq
        imp = improvement[mask]
        means.append(np.mean(imp))
        stds.append(np.std(imp) / np.sqrt(len(imp)) * 1.96)  # 95% CI
        counts.append(len(imp))

    x = np.arange(len(freq_order))
    colors = ['#2ca02c' if m > 2 else '#ff7f0e' if m > 0 else '#d62728' for m in means]
    bars = ax1.bar(x, means, color=colors, edgecolor='black', linewidth=0.5, width=0.7, alpha=0.85)
    ax1.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4, linewidth=1)

    for i, (m, n) in enumerate(zip(means, counts)):
        ax1.text(i, m + stds[i] + 0.8, f'n={n}', ha='center', fontsize=7, color='gray')

    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.axhline(y=np.mean(improvement), color='#9467bd', linewidth=1, linestyle='--',
                label=f'Overall mean: {np.mean(improvement):+.1f}%', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(freq_labels, rotation=30, ha='right')
    ax1.set_ylabel('Mean MASE improvement (%)')
    ax1.set_title('(a) Improvement by sampling frequency')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.15, axis='y')

    # Panel B: Scatter plot — improvement vs log(sampling period)
    for freq in freq_order:
        mask = freqs == freq
        period = FREQ_PERIOD[freq]
        imp = improvement[mask]
        ax2.scatter([np.log10(period)] * len(imp), imp, alpha=0.4, s=20, c='#1f77b4', zorder=2)

    # Add mean line
    freq_means_x = [np.log10(FREQ_PERIOD[f]) for f in freq_order]
    freq_means_y = means
    ax2.plot(freq_means_x, freq_means_y, 'o-', color='#d62728', markersize=6, linewidth=2,
             label='Frequency group mean', zorder=3)

    # Fit linear regression
    all_x = []
    all_y = []
    for freq in freq_order:
        mask = freqs == freq
        period = FREQ_PERIOD[freq]
        imp = improvement[mask]
        all_x.extend([np.log10(period)] * len(imp))
        all_y.extend(imp)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    slope, intercept = np.polyfit(all_x, all_y, 1)
    x_line = np.linspace(min(all_x), max(all_x), 100)
    ax2.plot(x_line, slope * x_line + intercept, '--', color='gray', linewidth=1,
             label=f'Trend: {slope:.1f}%/decade')

    # Correlation
    corr = np.corrcoef(all_x, all_y)[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes, fontsize=9,
             va='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('log$_{10}$(sampling period in seconds)')
    ax2.set_ylabel('MASE improvement (%)')
    ax2.set_title('(b) Improvement vs sampling period')
    ax2.legend(fontsize=7, loc='lower left')
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/frequency_correlation.pdf')
    plt.savefig(f'{FIGDIR}/frequency_correlation.png')
    print("Saved frequency_correlation.pdf/png")

    # Print correlation stats
    print(f"\nPearson r = {corr:.3f}")
    print(f"Slope = {slope:.2f}%/decade of sampling period")
    print(f"Intercept = {intercept:.2f}%")


if __name__ == '__main__':
    main()
