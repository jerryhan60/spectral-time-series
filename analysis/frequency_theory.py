"""
Frequency-dependent analysis: why hints help more at high frequencies.

Theory: patching at P=16 creates a blind spot at f > 1/P.
Chebyshev FIR hints fill this gap. Higher sampling rates have more
energy in the blind spot, so hints help more.

This figure connects the spectral theory to empirical results.
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
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
FIGDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"

def find_best_csv(target_mase):
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

bl_path = find_best_csv(1.2421)
ms_path = find_best_csv(1.1675)

bl_df = pd.read_csv(bl_path)
ms_df = pd.read_csv(ms_path)

mase_col = [c for c in bl_df.columns if 'MASE' in c and '0.5' in c]
if not mase_col:
    mase_col = [c for c in bl_df.columns if 'MASE' in c]
mase_col = mase_col[0]

bl_df['improvement_pct'] = (bl_df[mase_col] - ms_df[mase_col]) / bl_df[mase_col] * 100

# Map frequencies to approximate samples per day
freq_to_spd = {
    '5T': 288,
    '10S': 8640,
    '10T': 144,
    '15T': 96,
    'H': 24,
    'D': 1,
    'W': 1/7,
    'W-SUN': 1/7,
    'W-TUE': 1/7,
    'M': 1/30,
    'Q-DEC': 1/90,
    'Y': 1/365,
}

freq_order = ['10S', '5T', '10T', '15T', 'H', 'D', 'W', 'W-SUN', 'W-TUE', 'M', 'Q-DEC', 'Y']
freq_labels = ['10s', '5min', '10min', '15min', 'Hourly', 'Daily', 'Weekly', '', '', 'Monthly', 'Quarterly', 'Yearly']

bl_df['spd'] = bl_df['frequency'].map(freq_to_spd).fillna(1)

# Aggregate by frequency
freq_stats = bl_df.groupby('frequency').agg(
    n=('improvement_pct', 'count'),
    mean_imp=('improvement_pct', 'mean'),
    median_imp=('improvement_pct', 'median'),
    spd=('spd', 'first'),
).reset_index()

# Also get per-horizon breakdown within each frequency
horizon_freq = bl_df.groupby(['frequency', 'term']).agg(
    mean_imp=('improvement_pct', 'mean'),
    spd=('spd', 'first'),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Panel A: Improvement vs sampling frequency
ax = axes[0]
freq_stats_sorted = freq_stats.sort_values('spd', ascending=False)

# Log scale x-axis
valid = freq_stats_sorted[freq_stats_sorted['spd'] > 0]
colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in valid['mean_imp']]
ax.scatter(valid['spd'], valid['mean_imp'], c=colors, s=valid['n']*20,
           edgecolors='black', linewidth=0.5, zorder=5, alpha=0.8)

# Trend line
log_spd = np.log10(valid['spd'].values)
imp = valid['mean_imp'].values
z = np.polyfit(log_spd, imp, 1)
p = np.poly1d(z)
x_trend = np.logspace(np.log10(valid['spd'].min()), np.log10(valid['spd'].max()), 100)
ax.plot(x_trend, p(np.log10(x_trend)), 'k--', linewidth=1, alpha=0.5)

# Labels for key frequencies
for _, row in valid.iterrows():
    freq = row['frequency']
    if freq in ['10S', '5T', '15T', 'H', 'D', 'M', 'Q-DEC']:
        label = freq_labels[freq_order.index(freq)] if freq in freq_order else freq
        offset = (5, 5)
        ax.annotate(label, (row['spd'], row['mean_imp']), textcoords="offset points",
                   xytext=offset, fontsize=7, alpha=0.7)

ax.set_xscale('log')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle=':')
ax.set_xlabel('Sampling rate (samples/day)')
ax.set_ylabel('Mean MASE improvement (%)')
ax.set_title('(a) Improvement scales with sampling rate')
ax.grid(True, alpha=0.2)

# Add correlation annotation
from scipy import stats as sp_stats
r, p_val = sp_stats.pearsonr(log_spd, imp)
ax.text(0.05, 0.95, f'r = {r:.2f} (p = {p_val:.3f})', transform=ax.transAxes,
        fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: Theoretical spectral gap diagram
ax = axes[1]

# Show the "spectral budget" — what fraction of spectral energy is above 1/P
# For different frequencies, the Nyquist is at samples_per_day/2
# The patch blind spot starts at 1/(P * dt) = 1/(16 * dt)

# Theoretical illustration
P = 16
freqs_demo = np.linspace(0, 1, 1000)  # normalized frequency [0, 1] = [0, Nyquist]

# Energy distribution for different data types (schematic)
# Sub-hourly data: lots of high-freq energy
# Daily data: mostly low-freq energy
hf_energy = np.exp(-freqs_demo * 2)  # sub-hourly: broad spectrum
lf_energy = np.exp(-freqs_demo * 10)  # daily: concentrated at low freq

# Normalize
hf_energy /= hf_energy.sum()
lf_energy /= lf_energy.sum()

# Patch blind spot boundary
blind_spot = 1.0 / P  # fraction of Nyquist

ax.fill_between(freqs_demo, 0, hf_energy, alpha=0.3, color='#1f77b4', label='Sub-hourly spectrum')
ax.fill_between(freqs_demo, 0, lf_energy, alpha=0.3, color='#ff7f0e', label='Daily spectrum')
ax.plot(freqs_demo, hf_energy, color='#1f77b4', linewidth=1.5)
ax.plot(freqs_demo, lf_energy, color='#ff7f0e', linewidth=1.5)

# Mark blind spot
ax.axvline(x=blind_spot, color='red', linewidth=1.5, linestyle='--')
ax.axvspan(blind_spot, 1.0, alpha=0.1, color='red')
ax.text(blind_spot + 0.03, max(hf_energy) * 0.8, 'Patching\nblind spot',
        fontsize=8, color='red', va='top')

# Show energy in blind spot
hf_blind = hf_energy[freqs_demo >= blind_spot].sum() / hf_energy.sum() * 100
lf_blind = lf_energy[freqs_demo >= blind_spot].sum() / lf_energy.sum() * 100
ax.text(0.7, max(hf_energy) * 0.6,
        f'Sub-hourly:\n{hf_blind:.0f}% in\nblind spot',
        fontsize=7, color='#1f77b4', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(0.7, max(hf_energy) * 0.3,
        f'Daily:\n{lf_blind:.0f}% in\nblind spot',
        fontsize=7, color='#ff7f0e', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Normalized frequency (f / f_Nyquist)')
ax.set_ylabel('Spectral energy density')
ax.set_title('(b) Spectral gap mechanism')
ax.legend(loc='upper right', framealpha=0.8, fontsize=7)
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/frequency_theory.pdf')
plt.savefig(f'{FIGDIR}/frequency_theory.png')
print("Saved frequency_theory.pdf/png")

# Print frequency-level results
print("\nPer-frequency improvement:")
for _, row in freq_stats_sorted.iterrows():
    sign = '+' if row['mean_imp'] > 0 else ''
    print(f"  {row['frequency']:6s} ({int(row['n'])} configs): {sign}{row['mean_imp']:.1f}%")
