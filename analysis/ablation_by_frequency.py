"""
Ablation results broken down by frequency group.

Shows that the Chebyshev improvement is concentrated on high-frequency data,
while zero/random hints hurt uniformly.
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


def find_csv_by_mase(target_mase, tol=0.015):
    """Find CSV with geo mean closest to target."""
    all_csvs = glob.glob(os.path.join(RESULTS_DIR, "gifteval_results_epoch_*-step_10000_*.csv"))
    best_path, best_diff = None, 999
    for csv_path in all_csvs:
        try:
            df = pd.read_csv(csv_path)
            mase_col = [c for c in df.columns if 'MASE' in c]
            if not mase_col:
                continue
            mase = df[mase_col[0]].dropna()
            mase = mase[mase > 0]
            if len(mase) >= 95:
                gm = np.exp(np.mean(np.log(mase)))
                diff = abs(gm - target_mase)
                if diff < best_diff and diff < tol:
                    best_diff = diff
                    best_path = csv_path
        except Exception:
            continue
    return best_path


# Frequency groups from dataset names
FREQ_GROUPS = {
    'Sub-hourly': ['5T', '10T', '15T', '10S'],
    'Hourly': ['H'],
    'Daily+': ['D', 'W', 'M', 'Q'],
}

def get_freq_group(config_name):
    for group, freqs in FREQ_GROUPS.items():
        for freq in freqs:
            if f'/{freq}/' in config_name:
                return group
    # Fallback based on dataset name
    name_lower = config_name.lower()
    if any(x in name_lower for x in ['hourly', 'm4_hourly', 'weather_hourly']):
        return 'Hourly'
    if any(x in name_lower for x in ['quarterly', 'monthly', 'yearly', 'weekly']):
        return 'Daily+'
    return 'Daily+'


def load_and_compare(target_mase, name, baseline_df, mase_col):
    """Load CSV and compute per-config improvement vs baseline."""
    csv_path = find_csv_by_mase(target_mase)
    if csv_path is None:
        print(f"SKIP {name}: no CSV found for MASE={target_mase}")
        return None
    df = pd.read_csv(csv_path)
    mcol = [c for c in df.columns if 'MASE' in c][0]
    gm = np.exp(np.mean(np.log(df[mcol].dropna()[df[mcol].dropna() > 0])))
    print(f"Loaded {name}: geoMASE={gm:.4f} (target={target_mase}), file={os.path.basename(csv_path)}")
    return df[mcol].values


def main():
    # Load baseline
    bl_path = find_csv_by_mase(1.2421)
    if bl_path is None:
        print("No baseline CSV found!")
        return
    bl_df = pd.read_csv(bl_path)
    mase_col = [c for c in bl_df.columns if 'MASE' in c][0]
    bl_mase = bl_df[mase_col].values

    # Determine freq group per config
    config_col = None
    for col in ['dataset_config', 'config', 'dataset', 'Dataset']:
        if col in bl_df.columns:
            config_col = col
            break
    if config_col is None:
        # Use index-based assignment from known dataset ordering
        print("No config column found, using default frequency mapping")
        return

    freq_groups = [get_freq_group(str(c)) for c in bl_df[config_col].values]
    freq_arr = np.array(freq_groups)

    # Load ablation results
    methods = {
        'MS46': 1.1675,
        'Zero hints': 1.3201,
        'Random hints': 1.2806,
        'Duplicate': 1.2342,
    }

    results = {}
    for name, target in methods.items():
        data = load_and_compare(target, name, bl_df, mase_col)
        if data is not None:
            results[name] = data

    if not results:
        print("No ablation CSVs found!")
        return

    # Compute per-frequency-group improvement
    fig, ax = plt.subplots(figsize=(10, 5))

    groups_ordered = ['Sub-hourly', 'Hourly', 'Daily+']
    x = np.arange(len(groups_ordered))
    width = 0.18
    n_methods = len(results)
    offsets = np.linspace(-width * (n_methods - 1) / 2, width * (n_methods - 1) / 2, n_methods)

    colors = {
        'MS46': '#2ca02c',
        'Zero hints': '#d62728',
        'Random hints': '#ff7f0e',
        'Duplicate': '#9467bd',
    }

    for i, (name, data) in enumerate(results.items()):
        improvements = []
        counts = []
        for group in groups_ordered:
            mask = freq_arr == group
            if mask.sum() > 0:
                bl_g = bl_mase[mask]
                m_g = data[mask]
                imp = np.mean((bl_g - m_g) / bl_g * 100)
                improvements.append(imp)
                counts.append(mask.sum())
            else:
                improvements.append(0)
                counts.append(0)

        bars = ax.bar(x + offsets[i], improvements, width, label=name,
                      color=colors.get(name, '#333'), edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups_ordered)
    ax.set_ylabel('Mean MASE Improvement (%)')
    ax.set_title('Ablation Results by Frequency Group')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.15, axis='y')

    # Add count annotations
    for j, group in enumerate(groups_ordered):
        n = np.sum(freq_arr == group)
        ax.text(j, ax.get_ylim()[0] + 0.5, f'n={n}', ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/ablation_by_frequency.pdf')
    plt.savefig(f'{FIGDIR}/ablation_by_frequency.png')
    print("Saved ablation_by_frequency.pdf/png")


if __name__ == '__main__':
    main()
