#!/usr/bin/env python3
"""
Effect size analysis and forest plot for polynomial hint preconditioning.

Creates:
1. Forest plot with per-horizon and per-domain CIs (publication Figure)
2. Multi-metric agreement analysis (MASE vs sMAPE)
3. Effect size table (Cohen's d, Cliff's delta)

Usage:
    python analysis/effect_size_analysis.py
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
TAB_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

BL_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260219_121557.csv")
MS_CSV = os.path.join(RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260223_163428.csv")


def load_paired(bl_csv, ms_csv):
    bl = pd.read_csv(bl_csv)
    ms = pd.read_csv(ms_csv)

    mase_col = [c for c in bl.columns if 'MASE' in c and '0.5' in c]
    if not mase_col:
        mase_col = [c for c in bl.columns if 'MASE' in c]
    mase_col = mase_col[0]

    smape_col = [c for c in bl.columns if 'sMAPE' in c or 'SMAPE' in c]
    smape_col = smape_col[0] if smape_col else None

    bl = bl.rename(columns={mase_col: 'bl_mase'})
    ms = ms.rename(columns={mase_col: 'ms_mase'})

    keys = ['dataset', 'term']
    merged = bl[keys + ['bl_mase', 'frequency']].merge(
        ms[keys + ['ms_mase']], on=keys, how='inner'
    )

    if smape_col:
        bl_s = bl.rename(columns={smape_col: 'bl_smape'})
        ms_s = ms.rename(columns={smape_col: 'ms_smape'})
        merged = merged.merge(bl_s[keys + ['bl_smape']], on=keys, how='left')
        merged = merged.merge(ms_s[keys + ['ms_smape']], on=keys, how='left')

    valid = merged.dropna(subset=['bl_mase', 'ms_mase'])
    valid = valid[(valid['bl_mase'] > 0) & (valid['ms_mase'] > 0)]
    return valid


def cohens_d(a, b):
    """Paired Cohen's d."""
    diff = a - b
    return np.mean(diff) / np.std(diff, ddof=1)


def cliffs_delta(a, b):
    """Cliff's delta: non-parametric effect size."""
    n = len(a)
    more = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    return (more - less) / (n * n)


def bootstrap_ci(a, b, n_boot=10000, seed=42):
    """Bootstrap CI for geometric mean ratio."""
    rng = np.random.RandomState(seed)
    log_diff = np.log(a) - np.log(b)
    n = len(log_diff)
    boot_means = np.array([np.mean(log_diff[rng.choice(n, n, replace=True)]) for _ in range(n_boot)])
    return {
        'mean': np.exp(np.mean(log_diff)),
        'ci_low': np.exp(np.percentile(boot_means, 2.5)),
        'ci_high': np.exp(np.percentile(boot_means, 97.5)),
        'pct_change': (np.exp(np.mean(log_diff)) - 1) * 100,
        'ci_low_pct': (np.exp(np.percentile(boot_means, 2.5)) - 1) * 100,
        'ci_high_pct': (np.exp(np.percentile(boot_means, 97.5)) - 1) * 100,
    }


def categorize_domain(dataset_name):
    """Map dataset name to domain category."""
    ds = dataset_name.lower()
    if any(x in ds for x in ['solar', 'wind', 'elec']):
        return 'Energy'
    if any(x in ds for x in ['traffic', 'taxi', 'uber', 'rideshare', 'pedestrian', 'loop']):
        return 'Transport'
    if any(x in ds for x in ['bit', 'cloud']):
        return 'IT/Cloud'
    if any(x in ds for x in ['weather', 'temp']):
        return 'Weather'
    if any(x in ds for x in ['ett']):
        return 'ETT (Industrial)'
    if any(x in ds for x in ['m4', 'm1', 'm3', 'tourism']):
        return 'Competition'
    if any(x in ds for x in ['hospital', 'covid', 'nn5', 'cif']):
        return 'Misc'
    return 'Other'


def forest_plot(data):
    """Create forest plot with per-category CIs."""
    fig, ax = plt.subplots(figsize=(7, 8))

    categories = []

    # Overall
    overall = bootstrap_ci(data['ms_mase'].values, data['bl_mase'].values)
    wins = (data['ms_mase'] < data['bl_mase']).sum()
    categories.append(('Overall (n=97)', overall, wins, len(data), True))

    # By horizon
    for term in ['short', 'medium', 'long']:
        sub = data[data['term'] == term]
        if len(sub) >= 3:
            ci = bootstrap_ci(sub['ms_mase'].values, sub['bl_mase'].values)
            w = (sub['ms_mase'] < sub['bl_mase']).sum()
            categories.append((f'  {term.capitalize()} (n={len(sub)})', ci, w, len(sub), False))

    # Separator
    categories.append(('', None, 0, 0, False))

    # By domain
    data = data.copy()
    data['domain'] = data['dataset'].apply(categorize_domain)
    for domain in sorted(data['domain'].unique()):
        sub = data[data['domain'] == domain]
        if len(sub) >= 3:
            ci = bootstrap_ci(sub['ms_mase'].values, sub['bl_mase'].values)
            w = (sub['ms_mase'] < sub['bl_mase']).sum()
            categories.append((f'  {domain} (n={len(sub)})', ci, w, len(sub), False))

    # Separator
    categories.append(('', None, 0, 0, False))

    # By frequency
    freq_groups = {
        'Sub-hourly': ['5T', '10T', '15T', '30T'],
        'Hourly': ['H'],
        'Daily/Weekly': ['D', 'W'],
        'Monthly+': ['M', 'Q', 'Y', 'A'],
    }
    for grp_name, freqs in freq_groups.items():
        sub = data[data['frequency'].isin(freqs)]
        if len(sub) >= 3:
            ci = bootstrap_ci(sub['ms_mase'].values, sub['bl_mase'].values)
            w = (sub['ms_mase'] < sub['bl_mase']).sum()
            categories.append((f'  {grp_name} (n={len(sub)})', ci, w, len(sub), False))

    # Plot
    y_pos = []
    labels = []
    idx = 0
    for name, ci, wins, total, is_overall in reversed(categories):
        if name == '':
            idx += 0.5
            continue
        y_pos.append(idx)
        labels.append(name)
        if ci is not None:
            color = '#C44E52' if is_overall else '#4C72B0'
            marker = 'D' if is_overall else 'o'
            ms = 8 if is_overall else 6
            lw = 2.5 if is_overall else 1.5

            ax.errorbar(ci['pct_change'], idx,
                       xerr=[[ci['pct_change'] - ci['ci_low_pct']], [ci['ci_high_pct'] - ci['pct_change']]],
                       fmt=marker, color=color, markersize=ms, capsize=4, capthick=lw,
                       elinewidth=lw)

            # Win rate annotation
            ax.text(12, idx, f'{wins}/{total}', va='center', ha='left', fontsize=8,
                    fontweight='bold' if is_overall else 'normal')
        idx += 1

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Change in Geometric Mean MASE (%)', fontsize=10)
    ax.set_title('Forest Plot: MS $d{=}4{+}d{=}6$ vs Baseline', fontsize=12)

    # Win/loss header
    ax.text(12, max(y_pos) + 0.7, 'Win/n', va='center', ha='left', fontsize=8,
            fontweight='bold', fontstyle='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'forest_plot.{ext}'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved forest_plot.pdf/png")


def multi_metric_agreement(data):
    """Compare MASE and sMAPE improvements - shows robustness across metrics."""
    if 'bl_smape' not in data.columns or 'ms_smape' not in data.columns:
        print("sMAPE columns not found, skipping multi-metric analysis")
        return

    valid = data.dropna(subset=['bl_smape', 'ms_smape'])
    valid = valid[(valid['bl_smape'] > 0) & (valid['ms_smape'] > 0)]

    mase_imp = (valid['bl_mase'] - valid['ms_mase']) / valid['bl_mase'] * 100
    smape_imp = (valid['bl_smape'] - valid['ms_smape']) / valid['bl_smape'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): Scatter of MASE vs sMAPE improvement
    ax = axes[0]
    term_colors = {'short': '#55A868', 'medium': '#4C72B0', 'long': '#C44E52'}
    colors = [term_colors.get(t, 'gray') for t in valid['term']]
    ax.scatter(mase_imp, smape_imp, c=colors, alpha=0.6, s=25, edgecolors='black', linewidths=0.3)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.5)

    # Quadrant counts
    q1 = ((mase_imp > 0) & (smape_imp > 0)).sum()  # Both better
    q2 = ((mase_imp < 0) & (smape_imp > 0)).sum()  # MASE worse, sMAPE better
    q3 = ((mase_imp < 0) & (smape_imp < 0)).sum()  # Both worse
    q4 = ((mase_imp > 0) & (smape_imp < 0)).sum()  # MASE better, sMAPE worse

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1]*0.7, ylim[1]*0.85, f'{q1}', fontsize=14, fontweight='bold', color='green', ha='center')
    ax.text(xlim[0]*0.7, ylim[1]*0.85, f'{q2}', fontsize=14, fontweight='bold', color='orange', ha='center')
    ax.text(xlim[0]*0.7, ylim[0]*0.85, f'{q3}', fontsize=14, fontweight='bold', color='red', ha='center')
    ax.text(xlim[1]*0.7, ylim[0]*0.85, f'{q4}', fontsize=14, fontweight='bold', color='orange', ha='center')

    # Correlation
    corr = np.corrcoef(mase_imp, smape_imp)[0, 1]
    ax.set_xlabel('MASE Improvement (%)')
    ax.set_ylabel('sMAPE Improvement (%)')
    ax.set_title(f'(a) Metric Agreement ($r = {corr:.2f}$)')
    for t, c in term_colors.items():
        ax.scatter([], [], c=c, label=t.capitalize(), s=25, edgecolors='black', linewidths=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (b): Both metrics improvement by horizon
    ax = axes[1]
    horizons = ['short', 'medium', 'long']
    mase_by_h = []
    smape_by_h = []
    for h in horizons:
        sub = valid[valid['term'] == h]
        mase_geo = np.exp(np.mean(np.log(sub['ms_mase']))) / np.exp(np.mean(np.log(sub['bl_mase'])))
        smape_geo = np.exp(np.mean(np.log(sub['ms_smape']))) / np.exp(np.mean(np.log(sub['bl_smape'])))
        mase_by_h.append((mase_geo - 1) * 100)
        smape_by_h.append((smape_geo - 1) * 100)

    x = np.arange(len(horizons))
    width = 0.35
    ax.bar(x - width/2, mase_by_h, width, label='MASE', color='#4C72B0', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, smape_by_h, width, label='sMAPE', color='#E67E22', edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([h.capitalize() for h in horizons])
    ax.set_ylabel('Change (%)')
    ax.set_title('(b) Improvement by Horizon (both metrics)')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'multi_metric.{ext}'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved multi_metric.pdf/png")

    # Print summary
    agree = ((mase_imp > 0) & (smape_imp > 0)).sum() + ((mase_imp < 0) & (smape_imp < 0)).sum()
    print(f"\nMulti-metric agreement: {agree}/{len(valid)} = {agree/len(valid)*100:.0f}%")
    print(f"  Both improve: {q1}, both degrade: {q3}")
    print(f"  Disagreement: MASE worse/sMAPE better: {q2}, MASE better/sMAPE worse: {q4}")
    print(f"  Pearson r = {corr:.3f}")


def effect_size_table(data):
    """Generate LaTeX table of effect sizes."""
    rows = []

    # Overall
    log_ratio = np.log(data['ms_mase'].values / data['bl_mase'].values)
    d = cohens_d(data['bl_mase'].values, data['ms_mase'].values)
    cd = cliffs_delta(data['bl_mase'].values, data['ms_mase'].values)
    ci = bootstrap_ci(data['ms_mase'].values, data['bl_mase'].values)
    wins = (data['ms_mase'] < data['bl_mase']).sum()

    try:
        from scipy.stats import wilcoxon
        _, pval = wilcoxon(log_ratio)
    except ImportError:
        pval = None

    rows.append({
        'Category': 'Overall',
        'n': len(data),
        'wins': wins,
        'delta_pct': ci['pct_change'],
        'ci': f"[{ci['ci_low_pct']:.1f}, {ci['ci_high_pct']:.1f}]",
        'cohen_d': d,
        'cliff': cd,
        'p': pval,
    })

    # By horizon
    for term in ['short', 'medium', 'long']:
        sub = data[data['term'] == term]
        if len(sub) >= 3:
            lr = np.log(sub['ms_mase'].values / sub['bl_mase'].values)
            d = cohens_d(sub['bl_mase'].values, sub['ms_mase'].values)
            cd = cliffs_delta(sub['bl_mase'].values, sub['ms_mase'].values)
            ci = bootstrap_ci(sub['ms_mase'].values, sub['bl_mase'].values)
            w = (sub['ms_mase'] < sub['bl_mase']).sum()
            try:
                _, pval = wilcoxon(lr)
            except Exception:
                pval = None
            rows.append({
                'Category': term.capitalize(),
                'n': len(sub),
                'wins': w,
                'delta_pct': ci['pct_change'],
                'ci': f"[{ci['ci_low_pct']:.1f}, {ci['ci_high_pct']:.1f}]",
                'cohen_d': d,
                'cliff': cd,
                'p': pval,
            })

    # Print and save
    print("\n" + "=" * 80)
    print("EFFECT SIZE ANALYSIS")
    print("=" * 80)
    print(f"{'Category':>12} {'n':>4} {'Wins':>6} {'Delta%':>8} {'95% CI':>20} {'Cohen d':>9} {'Cliff d':>9} {'p-val':>10}")
    print("-" * 80)
    for r in rows:
        pstr = f"{r['p']:.2e}" if r['p'] is not None else "N/A"
        print(f"{r['Category']:>12} {r['n']:>4} {r['wins']:>6} {r['delta_pct']:>+7.1f}% {r['ci']:>20} {r['cohen_d']:>+9.3f} {r['cliff']:>+9.3f} {pstr:>10}")

    # LaTeX table
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Effect size analysis: MS $d{=}4{+}d{=}6$ vs baseline at 10K steps. Cohen's $d$ and Cliff's $\delta$ measure paired effect sizes; 95\% CIs from 10K bootstrap resamples.}")
    tex.append(r"\label{tab:effect_size}")
    tex.append(r"\begin{tabular}{lrrrrrrr}")
    tex.append(r"\toprule")
    tex.append(r"Category & $n$ & Wins & $\Delta$MASE\% & 95\% CI & Cohen's $d$ & Cliff's $\delta$ & $p$-value \\")
    tex.append(r"\midrule")
    for r in rows:
        pstr = f"${r['p']:.1e}$".replace("e-0", r"\times 10^{-").replace("e+0", r"\times 10^{") + "}" if r['p'] is not None else "---"
        bold = r"\textbf" if r['Category'] == 'Overall' else ""
        cat = f"{bold}{{{r['Category']}}}" if bold else r['Category']
        tex.append(f"{cat} & {r['n']} & {r['wins']} & {r['delta_pct']:+.1f}\\% & {r['ci']} & {r['cohen_d']:+.2f} & {r['cliff']:+.2f} & {pstr} \\\\")
        if r['Category'] == 'Overall':
            tex.append(r"\midrule")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    tex_path = os.path.join(TAB_DIR, "table_effect_size.tex")
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex))
    print(f"\nSaved {tex_path}")


def main():
    data = load_paired(BL_CSV, MS_CSV)
    print(f"Loaded {len(data)} paired configs")

    effect_size_table(data)
    forest_plot(data)
    multi_metric_agreement(data)


if __name__ == "__main__":
    main()
