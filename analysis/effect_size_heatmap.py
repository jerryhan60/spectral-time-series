"""
Effect size analysis by frequency and horizon.

Computes Cohen's d for MS46 vs baseline on log-MASE ratios,
broken down by sampling frequency and forecast horizon.
Creates a heatmap showing where the effect is strongest.
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

FREQ_PERIOD = {
    '10S': 10, '5T': 300, '10T': 600, '15T': 900,
    'H': 3600, 'D': 86400, 'W': 604800, 'M': 2592000,
}


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
            if abs(gm - target_mase) < best_diff and abs(gm - target_mase) < tol:
                best_diff = abs(gm - target_mase)
                best_path = csv_path
    return best_path


def main():
    bl_df = pd.read_csv(os.path.join(RESULTS_DIR, 'all_results_epoch_99-step_10000.csv'))
    ms_path = find_csv_by_mase(1.1675)
    ms_df = pd.read_csv(ms_path)
    mase_col = [c for c in ms_df.columns if 'MASE' in c][0]

    bl_mase = bl_df['MASE'].values
    ms_mase = ms_df[mase_col].values

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
    log_ratio = np.log(bl_mase) - np.log(ms_mase)

    freq_groups = {
        'Sub-hourly\n(10s-15min)': ['10S', '5T', '10T', '15T'],
        'Hourly': ['H'],
        'Daily+\n(D/W/M)': ['D', 'W', 'M', 'other'],
    }

    horizon_order = ['short', 'medium', 'long']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: Cohen's d heatmap
    heatmap_data = []
    fg_labels = list(freq_groups.keys())
    for fg_name, fg_freqs in freq_groups.items():
        row = []
        for h in horizon_order:
            mask = np.array([f in fg_freqs for f in freqs]) & (horizons == h)
            if mask.sum() >= 2:
                lr = log_ratio[mask]
                d = np.mean(lr) / np.std(lr) if np.std(lr) > 0 else 0
                row.append(d)
            else:
                row.append(np.nan)
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)
    im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.5)
    ax1.set_xticks(range(len(horizon_order)))
    ax1.set_xticklabels(['Short', 'Medium', 'Long'])
    ax1.set_yticks(range(len(fg_labels)))
    ax1.set_yticklabels(fg_labels)

    for i in range(len(fg_labels)):
        for j in range(len(horizon_order)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.8 else 'black'
                size_label = ''
                if abs(val) >= 0.8:
                    size_label = '\n(large)'
                elif abs(val) >= 0.5:
                    size_label = '\n(medium)'
                elif abs(val) >= 0.2:
                    size_label = '\n(small)'
                ax1.text(j, i, f'd={val:.2f}{size_label}', ha='center', va='center',
                         fontsize=8, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax1, shrink=0.8, label="Cohen's d")
    ax1.set_title("(a) Effect size: MS46 vs Baseline")
    ax1.set_xlabel('Forecast Horizon')

    # Panel B: Win rate heatmap
    winrate_data = []
    for fg_name, fg_freqs in freq_groups.items():
        row = []
        for h in horizon_order:
            mask = np.array([f in fg_freqs for f in freqs]) & (horizons == h)
            if mask.sum() > 0:
                wins = np.sum(ms_mase[mask] < bl_mase[mask])
                rate = wins / mask.sum() * 100
                row.append(rate)
            else:
                row.append(np.nan)
        winrate_data.append(row)

    winrate_data = np.array(winrate_data)
    im2 = ax2.imshow(winrate_data, cmap='RdYlGn', aspect='auto', vmin=30, vmax=100)
    ax2.set_xticks(range(len(horizon_order)))
    ax2.set_xticklabels(['Short', 'Medium', 'Long'])
    ax2.set_yticks(range(len(fg_labels)))
    ax2.set_yticklabels(fg_labels)

    for i in range(len(fg_labels)):
        for j in range(len(horizon_order)):
            val = winrate_data[i, j]
            if not np.isnan(val):
                mask = np.array([f in list(freq_groups.values())[i] for f in freqs]) & (horizons == horizon_order[j])
                n = mask.sum()
                wins = int(np.sum(ms_mase[mask] < bl_mase[mask]))
                color = 'white' if val > 80 else 'black'
                ax2.text(j, i, f'{wins}/{n}\n({val:.0f}%)', ha='center', va='center',
                         fontsize=8, color=color, fontweight='bold')

    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Win rate (%)')
    ax2.set_title("(b) Win rate: MS46 vs Baseline")
    ax2.set_xlabel('Forecast Horizon')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/effect_size_heatmap.pdf')
    plt.savefig(f'{FIGDIR}/effect_size_heatmap.png')
    print("Saved effect_size_heatmap.pdf/png")

    # Print summary
    print("\n--- Effect Size Summary ---")
    print(f"{'Group':20s} {'Horizon':>8s} {'n':>4s} {'Cohen d':>8s} {'Win rate':>10s} {'Mean imp':>10s}")
    print("-" * 65)
    for i, (fg_name, fg_freqs) in enumerate(freq_groups.items()):
        for j, h in enumerate(horizon_order):
            mask = np.array([f in fg_freqs for f in freqs]) & (horizons == h)
            n = mask.sum()
            if n >= 2:
                lr = log_ratio[mask]
                d = np.mean(lr) / np.std(lr) if np.std(lr) > 0 else 0
                wins = np.sum(ms_mase[mask] < bl_mase[mask])
                imp = np.mean((bl_mase[mask] - ms_mase[mask]) / bl_mase[mask] * 100)
                fg_clean = fg_name.replace('\n',' ')
                print(f"{fg_clean:20s} {h:>8s} {n:>4d} {d:>8.3f} {wins:>3d}/{n:<3d}({wins/n*100:4.0f}%) {imp:>+9.1f}%")


if __name__ == '__main__':
    main()
