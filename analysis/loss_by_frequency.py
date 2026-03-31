"""
Loss decomposition by frequency band.

Analyzes per-config GIFT-Eval results to show that hint improvement
varies systematically with data sampling frequency.
Creates publication-quality figure showing:
(a) Improvement distribution by frequency group
(b) Win rate by frequency group
(c) Correlation between frequency and improvement
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
try:
    from scipy import stats
except ImportError:
    stats = None

rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

FIGDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
RESULTS = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"

# Load baseline and ms46 results
bl_file = f"{RESULTS}/all_results_epoch_99-step_10000.csv"
# ms46 results should be a separate file
# Need to find the right result files

# Frequency mapping from dataset name patterns
FREQ_MAP = {
    '5T': ('5T', 288, 'Sub-hourly'),
    '10T': ('10T', 144, 'Sub-hourly'),
    '15T': ('15T', 96, 'Sub-hourly'),
    '10S': ('10S', 8640, 'Sub-hourly'),
    'H': ('H', 24, 'Hourly'),
    'D': ('D', 7, 'Daily'),
    'W': ('W', 1, 'Weekly+'),
    'M': ('M', 1/4, 'Weekly+'),
    'Q': ('Q', 1/12, 'Weekly+'),
    'QS': ('QS', 1/12, 'Weekly+'),
    'A': ('A', 1/52, 'Weekly+'),
}

def get_frequency_group(dataset_name):
    """Extract frequency group from dataset configuration name."""
    parts = dataset_name.split('/')
    if len(parts) >= 2:
        freq = parts[1]
        if freq in FREQ_MAP:
            return FREQ_MAP[freq]
    # Default based on common patterns
    for freq_key in ['10S', '5T', '10T', '15T']:
        if freq_key in dataset_name:
            return FREQ_MAP[freq_key]
    if '/H/' in dataset_name:
        return FREQ_MAP['H']
    if '/D/' in dataset_name:
        return FREQ_MAP['D']
    if '/W/' in dataset_name:
        return FREQ_MAP['W']
    if '/M/' in dataset_name or 'm4_monthly' in dataset_name.lower():
        return FREQ_MAP['M']
    if '/Q/' in dataset_name or 'quarterly' in dataset_name.lower():
        return FREQ_MAP['Q']
    return ('Unknown', 0, 'Unknown')


def load_results(csv_path):
    """Load GIFT-Eval results and compute per-config MASE."""
    df = pd.read_csv(csv_path)
    if 'MASE' not in df.columns:
        print(f"Warning: MASE not in {csv_path}")
        return None
    # Use dataset_config or similar column
    config_col = None
    for col in ['dataset_config', 'config', 'dataset', 'Dataset']:
        if col in df.columns:
            config_col = col
            break
    if config_col is None:
        print(f"Warning: No config column in {csv_path}")
        return None
    return df[[config_col, 'MASE']].rename(columns={config_col: 'config'})


def create_freq_improvement_figure():
    """Create frequency-dependent improvement figure from existing data."""
    # Use hardcoded data from experiment_summary.md
    freq_data = {
        'Sub-hourly (10S)': {'n': 6, 'improvement': 23.1, 'win_rate': 100},
        'Sub-hourly (15T)': {'n': 5, 'improvement': 13.3, 'win_rate': 100},
        'Sub-hourly (10T)': {'n': 2, 'improvement': 11.2, 'win_rate': 100},
        'Sub-hourly (5T)': {'n': 12, 'improvement': 5.6, 'win_rate': 83},
        'Hourly (H)': {'n': 31, 'improvement': 2.8, 'win_rate': 55},
        'Daily (D)': {'n': 15, 'improvement': -0.7, 'win_rate': 53},
        'Weekly+ (W/M/Q)': {'n': 12, 'improvement': -3.5, 'win_rate': 33},
    }

    # For the ms46 figure, use the per-horizon data
    horizon_data = {
        'Short': {'n': 55, 'improvement': 2.45, 'win_rate': 67.3, 'std': 6.5},
        'Medium': {'n': 21, 'improvement': 8.51, 'win_rate': 85.7, 'std': 5.2},
        'Long': {'n': 21, 'improvement': 12.41, 'win_rate': 90.5, 'std': 7.8},
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Improvement by frequency group
    ax = axes[0]
    freqs = list(freq_data.keys())
    improvements = [freq_data[f]['improvement'] for f in freqs]
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in improvements]
    bars = ax.barh(range(len(freqs)), improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels(freqs, fontsize=8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('MASE Improvement (%)')
    ax.set_title('(a) Improvement by Frequency')
    ax.invert_yaxis()
    for i, (v, n) in enumerate(zip(improvements, [freq_data[f]['n'] for f in freqs])):
        ax.text(v + 0.5 if v >= 0 else v - 0.5, i, f'n={n}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=7)

    # (b) Win rate by frequency
    ax = axes[1]
    win_rates = [freq_data[f]['win_rate'] for f in freqs]
    colors_wr = ['#2ca02c' if v > 50 else '#d62728' for v in win_rates]
    ax.barh(range(len(freqs)), win_rates, color=colors_wr, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels(freqs, fontsize=8)
    ax.axvline(x=50, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Win Rate (%)')
    ax.set_title('(b) Win Rate by Frequency')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    # (c) Improvement by horizon
    ax = axes[2]
    horizons = list(horizon_data.keys())
    h_imp = [horizon_data[h]['improvement'] for h in horizons]
    h_std = [horizon_data[h]['std'] for h in horizons]
    h_win = [horizon_data[h]['win_rate'] for h in horizons]
    x = np.arange(len(horizons))
    bars = ax.bar(x, h_imp, yerr=h_std, color=['#2ca02c']*3, alpha=0.8,
                  edgecolor='black', linewidth=0.5, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel('MASE Improvement (%)')
    ax.set_title('(c) Improvement by Horizon')
    ax.axhline(y=0, color='black', linewidth=0.5)
    for i, (v, wr) in enumerate(zip(h_imp, h_win)):
        ax.text(i, v + h_std[i] + 0.5, f'{wr:.0f}% win', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/freq_horizon_improvement.pdf')
    plt.savefig(f'{FIGDIR}/freq_horizon_improvement.png')
    print("Saved freq_horizon_improvement.pdf/png")


if __name__ == '__main__':
    create_freq_improvement_figure()
