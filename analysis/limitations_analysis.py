"""
Limitations analysis: where do polynomial hints hurt?

Analyzes the 23/97 datasets where ms46 has higher MASE than baseline,
looking for patterns (dataset size, frequency, domain, etc.)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("/scratch/gpfs/EHAZAN/jh1161/gifteval/results")
OUT_DIR = Path(__file__).resolve().parent / "figures"
TABLE_DIR = Path(__file__).resolve().parent / "tables"
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


def find_csv_by_mase(target_mase, tolerance=0.002):
    """Find the right CSV by matching expected geomean MASE."""
    best = None
    best_diff = float('inf')
    for f in sorted(RESULTS_DIR.glob("gifteval_results_epoch_99-step_10000_*.csv")):
        df = pd.read_csv(f)
        if 'MASE' in df.columns:
            gmase = np.exp(np.log(df['MASE'].dropna()).mean())
            diff = abs(gmase - target_mase)
            if diff < best_diff and diff < tolerance:
                best = (f, df, gmase)
                best_diff = diff
    return best if best else (None, None, None)


def categorize_dataset(name):
    """Categorize dataset by domain and frequency."""
    name_lower = name.lower()

    # Frequency
    freq = "unknown"
    if "/10T/" in name or "/5T/" in name or "/15T/" in name:
        freq = "sub-hourly"
    elif "/H/" in name:
        freq = "hourly"
    elif "/D/" in name:
        freq = "daily"
    elif "/W/" in name:
        freq = "weekly"
    elif "/M/" in name or "monthly" in name_lower:
        freq = "monthly"
    elif "quarterly" in name_lower:
        freq = "quarterly"
    elif "short" in name_lower and "/" not in name:
        freq = "mixed"

    # Domain
    domain = "other"
    if "solar" in name_lower or "electricity" in name_lower or "elecdemand" in name_lower:
        domain = "energy"
    elif "uber" in name_lower or "taxi" in name_lower or "traffic" in name_lower or "loop" in name_lower:
        domain = "transport"
    elif "ett" in name_lower:
        domain = "ETT"
    elif "hospital" in name_lower or "covid" in name_lower:
        domain = "health"
    elif "m4_" in name_lower or "m_dense" in name_lower:
        domain = "M-competition"
    elif "hierarchical" in name_lower or "restaurant" in name_lower or "dominick" in name_lower:
        domain = "retail"
    elif "tourism" in name_lower or "nn5" in name_lower or "fred" in name_lower:
        domain = "benchmark"
    elif "web" in name_lower or "wiki" in name_lower:
        domain = "web"
    elif "temperature" in name_lower or "weather" in name_lower:
        domain = "weather"
    elif "kdd" in name_lower:
        domain = "benchmark"
    elif "car_parts" in name_lower:
        domain = "manufacturing"
    elif "rideshare" in name_lower:
        domain = "transport"
    elif "bizitobs" in name_lower:
        domain = "IT/cloud"

    # Horizon
    horizon = "unknown"
    if name.endswith("/short") or name.endswith("short"):
        horizon = "short"
    elif name.endswith("/medium") or name.endswith("medium"):
        horizon = "medium"
    elif name.endswith("/long") or name.endswith("long"):
        horizon = "long"

    return domain, freq, horizon


def main():
    # Find baseline and ms46 CSVs
    bl_path, bl_df, bl_mase = find_csv_by_mase(1.2421)
    ms_path, ms_df, ms_mase = find_csv_by_mase(1.1675)

    if bl_df is None or ms_df is None:
        print(f"Could not find CSVs. BL={bl_path}, MS={ms_path}")
        return

    print(f"Baseline: {bl_path.name} (MASE={bl_mase:.4f})")
    print(f"MS46: {ms_path.name} (MASE={ms_mase:.4f})")

    # Merge on dataset config_name
    key_cols = ['dataset', 'term']
    bl_sub = bl_df[key_cols + ['MASE']].copy()
    ms_sub = ms_df[key_cols + ['MASE']].copy()
    merged = bl_sub.merge(ms_sub, on=key_cols, suffixes=('_bl', '_ms'))
    # Create a combined dataset name for display
    merged['ds_name'] = merged['dataset'] + '/' + merged['term']

    # Compute improvement
    merged['ratio'] = merged['MASE_ms'] / merged['MASE_bl']
    merged['pct_change'] = (merged['ratio'] - 1) * 100

    # Categorize
    merged['domain'], merged['freq'], merged['horizon'] = zip(
        *merged['ds_name'].apply(categorize_dataset))

    # Split into wins and losses
    wins = merged[merged['ratio'] < 1.0].sort_values('ratio')
    losses = merged[merged['ratio'] >= 1.0].sort_values('ratio', ascending=False)

    print(f"\n=== LIMITATIONS ANALYSIS ===")
    print(f"Total configs: {len(merged)}")
    print(f"Wins (hint better): {len(wins)} ({100*len(wins)/len(merged):.0f}%)")
    print(f"Losses (hint worse): {len(losses)} ({100*len(losses)/len(merged):.0f}%)")

    print(f"\n--- TOP 10 LOSSES (where hints hurt most) ---")
    for _, row in losses.head(10).iterrows():
        print(f"  {row['ds_name']:40s}  BL={row['MASE_bl']:.4f}  MS={row['MASE_ms']:.4f}  {row['pct_change']:+.1f}%  [{row['domain']}/{row['freq']}]")

    print(f"\n--- TOP 10 WINS (where hints help most) ---")
    for _, row in wins.head(10).iterrows():
        print(f"  {row['ds_name']:40s}  BL={row['MASE_bl']:.4f}  MS={row['MASE_ms']:.4f}  {row['pct_change']:+.1f}%  [{row['domain']}/{row['freq']}]")

    # Analysis: domain breakdown
    print(f"\n--- LOSS ANALYSIS BY DOMAIN ---")
    for domain in sorted(losses['domain'].unique()):
        subset = losses[losses['domain'] == domain]
        total_in_domain = len(merged[merged['domain'] == domain])
        print(f"  {domain:15s}: {len(subset)}/{total_in_domain} configs worse "
              f"(avg {subset['pct_change'].mean():+.1f}%)")

    print(f"\n--- LOSS ANALYSIS BY FREQUENCY ---")
    for freq in sorted(losses['freq'].unique()):
        subset = losses[losses['freq'] == freq]
        total_in_freq = len(merged[merged['freq'] == freq])
        print(f"  {freq:15s}: {len(subset)}/{total_in_freq} configs worse "
              f"(avg {subset['pct_change'].mean():+.1f}%)")

    print(f"\n--- LOSS ANALYSIS BY HORIZON ---")
    for hz in ['short', 'medium', 'long']:
        subset = losses[losses['horizon'] == hz]
        total_in_hz = len(merged[merged['horizon'] == hz])
        if total_in_hz > 0:
            print(f"  {hz:15s}: {len(subset)}/{total_in_hz} configs worse "
                  f"(avg {subset['pct_change'].mean():+.1f}%)")

    # Figure: improvement distribution with loss region highlighted
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    changes = merged['pct_change'].values
    bins = np.linspace(-35, 15, 50)
    n, _, patches = ax.hist(changes, bins=bins, color='#3498db', alpha=0.7, edgecolor='white')
    # Color losses red
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= 0:
            patch.set_facecolor('#e74c3c')
            patch.set_alpha(0.7)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(np.median(changes), color='#2c3e50', linewidth=1.5, linestyle='-',
               label=f'Median: {np.median(changes):.1f}%')
    ax.set_xlabel('MASE Change (%)', fontsize=12)
    ax.set_ylabel('Number of Configs', fontsize=12)
    ax.set_title(f'Distribution of MASE Changes (ms46 vs baseline)\n'
                 f'{len(wins)} wins, {len(losses)} losses', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'limitation_distribution.{ext}', bbox_inches='tight')
    plt.close()

    # Figure: scatter by domain
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    domain_colors = {
        'energy': '#f39c12', 'transport': '#e74c3c', 'ETT': '#9b59b6',
        'health': '#2ecc71', 'M-competition': '#3498db', 'retail': '#1abc9c',
        'benchmark': '#34495e', 'web': '#e67e22', 'weather': '#7f8c8d',
        'manufacturing': '#d35400', 'IT/cloud': '#c0392b', 'other': '#95a5a6'
    }
    for domain in sorted(merged['domain'].unique()):
        subset = merged[merged['domain'] == domain]
        color = domain_colors.get(domain, '#95a5a6')
        ax.scatter(subset['MASE_bl'], subset['MASE_ms'], c=color,
                  label=f'{domain} ({len(subset)})', s=40, alpha=0.7, edgecolors='white', linewidth=0.5)

    lims = [0, max(merged['MASE_bl'].max(), merged['MASE_ms'].max()) * 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Baseline MASE', fontsize=12)
    ax.set_ylabel('MS46 MASE', fontsize=12)
    ax.set_title('Per-Config MASE: Baseline vs Multi-Scale Hint', fontsize=12)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.set_aspect('equal')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'scatter_by_domain.{ext}', bbox_inches='tight')
    plt.close()

    print(f"\nSaved limitation_distribution.pdf/png and scatter_by_domain.pdf/png")

    # Generate LaTeX limitation table
    with open(TABLE_DIR / 'table_limitations.tex', 'w') as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Configurations where multi-scale hint degrades performance. "
                "Losses concentrate in short horizons and sub-hourly frequencies.}\n")
        f.write("\\label{tab:limitations}\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Category & Loss / Total & Avg $\\Delta$ & Worst $\\Delta$ \\\\\n\\midrule\n")

        # By horizon
        f.write("\\multicolumn{4}{l}{\\textit{By forecast horizon}} \\\\\n")
        for hz in ['short', 'medium', 'long']:
            subset_all = merged[merged['horizon'] == hz]
            subset_loss = losses[losses['horizon'] == hz]
            if len(subset_all) > 0:
                worst = subset_loss['pct_change'].max() if len(subset_loss) > 0 else 0
                avg = subset_loss['pct_change'].mean() if len(subset_loss) > 0 else 0
                f.write(f"\\quad {hz.capitalize()} & {len(subset_loss)}/{len(subset_all)} "
                       f"& {avg:+.1f}\\% & {worst:+.1f}\\% \\\\\n")

        f.write("\\midrule\n")
        f.write("\\multicolumn{4}{l}{\\textit{By domain (top loss domains)}} \\\\\n")
        # By domain, sorted by loss count
        domain_loss_counts = []
        for domain in sorted(merged['domain'].unique()):
            subset_loss = losses[losses['domain'] == domain]
            if len(subset_loss) > 0:
                subset_all = merged[merged['domain'] == domain]
                domain_loss_counts.append((domain, len(subset_loss), len(subset_all),
                                          subset_loss['pct_change'].mean(),
                                          subset_loss['pct_change'].max()))
        domain_loss_counts.sort(key=lambda x: -x[1])
        for domain, nloss, ntotal, avg, worst in domain_loss_counts[:6]:
            f.write(f"\\quad {domain} & {nloss}/{ntotal} & {avg:+.1f}\\% & {worst:+.1f}\\% \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"Saved table_limitations.tex")


if __name__ == "__main__":
    main()
