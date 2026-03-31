"""
Ablation results broken down by forecast horizon.

Shows that:
- MS46 Chebyshev improvement scales with horizon (short < medium < long)
- Zero/random hints hurt uniformly across all horizons
- Duplicate/learned show no consistent horizon pattern
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


def main():
    # Load baseline
    bl_df = pd.read_csv(os.path.join(RESULTS_DIR, 'all_results_epoch_99-step_10000.csv'))
    mase_col = 'MASE'
    bl_mase = bl_df[mase_col].values

    # Get horizon info
    if 'term' not in bl_df.columns:
        # Parse from dataset_config
        configs = bl_df['dataset_config'].values
        horizons = []
        for c in configs:
            parts = c.split('/')
            h = parts[-1] if parts[-1] in ['short', 'medium', 'long'] else 'short'
            horizons.append(h)
        bl_df['term'] = horizons

    horizons_arr = bl_df['term'].values

    methods = {
        'MS46 (Cheb d=4+d=6)': 1.1675,
        'Duplicate input': 1.2342,
        'Learned 4-tap': 1.2351,
        'Learned 16-tap': 1.2775,
        'Zero hints': 1.3201,
        'Random hints': 1.2806,
    }

    # Load all ablation results
    method_data = {}
    for name, target in methods.items():
        path = find_csv_by_mase(target)
        if path:
            df = pd.read_csv(path)
            mcol = [c for c in df.columns if 'MASE' in c][0]
            method_data[name] = df[mcol].values
            gm = np.exp(np.mean(np.log(df[mcol].dropna()[df[mcol].dropna() > 0])))
            print(f"Loaded {name}: geoMASE={gm:.4f} (target={target})")
        else:
            print(f"SKIP {name}: no CSV found")

    if not method_data:
        print("No data loaded!")
        return

    # --- Figure: ablation by horizon ---
    fig, ax = plt.subplots(figsize=(11, 5))

    horizon_order = ['short', 'medium', 'long']
    horizon_labels = ['Short (n=55)', 'Medium (n=21)', 'Long (n=21)']
    x = np.arange(len(horizon_order))
    n_methods = len(method_data)
    width = 0.12
    offsets = np.linspace(-width * (n_methods - 1) / 2, width * (n_methods - 1) / 2, n_methods)

    colors = {
        'MS46 (Cheb d=4+d=6)': '#2ca02c',
        'Duplicate input': '#9467bd',
        'Learned 4-tap': '#8c564b',
        'Learned 16-tap': '#e377c2',
        'Zero hints': '#d62728',
        'Random hints': '#ff7f0e',
    }

    for i, (name, data) in enumerate(method_data.items()):
        improvements = []
        for h in horizon_order:
            mask = horizons_arr == h
            if mask.sum() > 0:
                bl_h = bl_mase[mask]
                m_h = data[mask]
                imp = np.mean((bl_h - m_h) / bl_h * 100)
                improvements.append(imp)
            else:
                improvements.append(0)

        bars = ax.bar(x + offsets[i], improvements, width, label=name,
                      color=colors.get(name, '#333'), edgecolor='black', linewidth=0.4, alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_labels, fontsize=10)
    ax.set_ylabel('Mean MASE Improvement vs Baseline (%)')
    ax.set_title('Ablation Results by Forecast Horizon')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.15, axis='y')

    # Add annotation
    ax.annotate('Chebyshev scales\nwith horizon', xy=(2, 10), fontsize=8,
                color='#2ca02c', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/ablation_horizon.pdf')
    plt.savefig(f'{FIGDIR}/ablation_horizon.png')
    print("Saved ablation_horizon.pdf/png")

    # Print detailed table
    print("\n--- Per-Horizon Ablation Summary ---")
    print(f"{'Method':30s} {'Short':>8s} {'Medium':>8s} {'Long':>8s} {'Overall':>8s}")
    print("-" * 60)
    for name, data in method_data.items():
        vals = []
        for h in horizon_order:
            mask = horizons_arr == h
            bl_h = bl_mase[mask]
            m_h = data[mask]
            imp = np.mean((bl_h - m_h) / bl_h * 100)
            vals.append(imp)
        overall = np.mean((bl_mase - data) / bl_mase * 100)
        print(f"{name:30s} {vals[0]:+7.1f}% {vals[1]:+7.1f}% {vals[2]:+7.1f}% {overall:+7.1f}%")


if __name__ == '__main__':
    main()
