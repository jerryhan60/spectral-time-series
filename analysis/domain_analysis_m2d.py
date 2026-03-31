"""
Per-domain and per-frequency breakdown of GIFT-Eval results (m2d, 5 seeds).

Uses all 5 seeds of HD10 vs Baseline (lotsa_v1_moirai2 data) to produce
publication-quality domain/frequency breakdown with statistical significance.

Outputs:
  - analysis/figures/domain_improvement_m2d.pdf
  - analysis/figures/frequency_improvement_m2d.pdf
  - analysis/tables/table_domain_m2d.tex
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gmean, binomtest

RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
TABLE_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# 5-seed CSV mappings (m2d = lotsa_v1_moirai2 data, 10K steps)
# VERIFIED from SLURM eval logs (core_m2d, seeds_m2d, ev_s7m2, etc.)
SEED_CSVS = {
    'BL': {
        0: "gifteval_results_epoch_99-step_10000_20260308_000815.csv",   # 1.2185
        1: "gifteval_results_epoch_99-step_10000_20260308_002546.csv",   # 1.1955
        2: "gifteval_results_epoch_99-step_10000_20260308_030127.csv",   # 1.2134
        7: "gifteval_results_epoch_99-step_10000_20260308_190028.csv",   # 1.2198
        42: "gifteval_results_epoch_99-step_10000_20260309_011307.csv",  # 1.2111
    },
    'HD10': {
        0: "gifteval_results_epoch_99-step_10000_20260308_005524.csv",   # 1.1577
        1: "gifteval_results_epoch_99-step_10000_20260308_011431.csv",   # 1.1902
        2: "gifteval_results_epoch_99-step_10000_20260308_034842.csv",   # 1.1798
        7: "gifteval_results_epoch_99-step_10000_20260308_194312.csv",   # 1.1698
        42: "gifteval_results_epoch_99-step_10000_20260309_033157.csv",  # 1.1852
    },
    'MS46': {
        0: "gifteval_results_epoch_99-step_10000_20260307_200919.csv",   # 1.1999
        1: "gifteval_results_epoch_99-step_10000_20260308_050016.csv",   # 1.1792
        2: "gifteval_results_epoch_99-step_10000_20260308_054233.csv",   # 1.1711
        7: "gifteval_results_epoch_99-step_10000_20260308_213747.csv",   # 1.2119
        42: "gifteval_results_epoch_99-step_10000_20260309_015542.csv",  # 1.1670
    },
    'MSHD10': {
        # Seeds 0-2: mshd10_m2d_seed*; Seeds 7,42: ms46hd10_m2d_seed*
        0: "gifteval_results_epoch_99-step_10000_20260308_010128.csv",   # 1.2029
        1: "gifteval_results_epoch_99-step_10000_20260308_015158.csv",   # 1.1729
        2: "gifteval_results_epoch_99-step_10000_20260308_033919.csv",   # 1.1876
        7: "gifteval_results_epoch_99-step_10000_20260309_042201.csv",   # 1.1885
        42: "gifteval_results_epoch_99-step_10000_20260309_043555.csv",  # 1.1794
    },
}

DOMAIN_MAP = {
    "electricity": "Energy",
    "solar": "Energy",
    "LOOP_SEATTLE": "Transport",
    "SZ_TAXI": "Transport",
    "jena_weather": "Weather",
    "temperature_rain_with_missing": "Weather",
    "kdd_cup_2018_with_missing": "Weather",
    "saugeenday": "Weather",
    "ett1": "ETT (Industrial)",
    "ett2": "ETT (Industrial)",
    "bitbrains_fast_storage": "IT/Cloud",
    "bitbrains_rnd": "IT/Cloud",
    "bizitobs_application": "IT/Cloud",
    "bizitobs_service": "IT/Cloud",
    "bizitobs_l2c": "IT/Cloud",
    "M_DENSE": "IT/Cloud",
    "us_births": "Demographics",
    "covid_deaths": "Healthcare",
    "hospital": "Healthcare",
    "car_parts_with_missing": "Retail/Sales",
    "hierarchical_sales": "Retail/Sales",
    "restaurant": "Retail/Sales",
    "m4_daily": "M4 Competition",
    "m4_hourly": "M4 Competition",
    "m4_monthly": "M4 Competition",
    "m4_quarterly": "M4 Competition",
    "m4_weekly": "M4 Competition",
    "m4_yearly": "M4 Competition",
}

FREQ_ORDER = {"sub-hourly": 0, "hourly": 1, "daily": 2, "weekly": 3, "monthly": 4, "quarterly": 5, "yearly": 6}


def classify_freq(raw_freq: str) -> str:
    f = raw_freq.strip()
    if f in ("5T", "10T", "15T", "10S"):
        return "sub-hourly"
    if f == "H":
        return "hourly"
    if f == "D":
        return "daily"
    if f.startswith("W"):
        return "weekly"
    if f == "M":
        return "monthly"
    if f.startswith("Q"):
        return "quarterly"
    if f.startswith("A"):
        return "yearly"
    return f


def load_seed_data(model_name):
    """Load all seeds for a model, return DataFrame with config as key."""
    frames = []
    for seed, csv_name in SEED_CSVS[model_name].items():
        path = os.path.join(RESULTS_DIR, csv_name)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping seed {seed}")
            continue
        df = pd.read_csv(path)
        df["config"] = df["dataset"] + "/" + df["term"]
        df["seed"] = seed
        frames.append(df[["config", "dataset", "term", "frequency", "MASE", "seed"]])
    return pd.concat(frames, ignore_index=True)


def main():
    # Load all data
    bl_data = load_seed_data('BL')
    hd10_data = load_seed_data('HD10')
    ms46_data = load_seed_data('MS46')
    mshd10_data = load_seed_data('MSHD10')

    # Verify
    for name, data in [('BL', bl_data), ('HD10', hd10_data), ('MS46', ms46_data), ('MSHD10', mshd10_data)]:
        n_seeds = data['seed'].nunique()
        n_configs = data['config'].nunique()
        print(f"{name}: {n_seeds} seeds, {n_configs} configs, {len(data)} rows")

    # Pool all seeds: merge BL and HD10 by (config, seed)
    merged = bl_data.merge(hd10_data[["config", "seed", "MASE"]], on=["config", "seed"], suffixes=("_bl", "_hd10"))
    merged["base_dataset"] = merged["dataset"].apply(lambda x: x.split("/")[0])
    merged["domain"] = merged["base_dataset"].map(DOMAIN_MAP).fillna("Other")
    merged["freq_bucket"] = merged["frequency"].apply(classify_freq)
    merged["improvement_pct"] = (1 - merged["MASE_hd10"] / merged["MASE_bl"]) * 100
    merged["hd10_wins"] = merged["MASE_hd10"] < merged["MASE_bl"]

    print(f"\nPooled dataset: {len(merged)} rows ({merged['seed'].nunique()} seeds x {merged['config'].nunique()} configs)")
    overall_gm_bl = gmean(merged["MASE_bl"])
    overall_gm_hd = gmean(merged["MASE_hd10"])
    overall_wins = merged["hd10_wins"].sum()
    overall_n = len(merged)
    overall_p = binomtest(int(overall_wins), overall_n, 0.5, alternative='greater').pvalue
    print(f"Overall: BL={overall_gm_bl:.4f}, HD10={overall_gm_hd:.4f}, "
          f"Δ={(1 - overall_gm_hd/overall_gm_bl)*100:+.2f}%, "
          f"wins={overall_wins}/{overall_n} ({overall_wins/overall_n*100:.1f}%), p={overall_p:.2e}")

    # Group stats
    def group_stats(df, group_col):
        rows = []
        for grp, sub in df.groupby(group_col):
            n = len(sub)
            geo_bl = gmean(sub["MASE_bl"])
            geo_hd = gmean(sub["MASE_hd10"])
            imp = (1 - geo_hd / geo_bl) * 100
            wins = int(sub["hd10_wins"].sum())
            p = binomtest(wins, n, 0.5, alternative='greater').pvalue
            rows.append({"group": grp, "n": n, "geo_bl": geo_bl, "geo_hd": geo_hd,
                         "improvement_pct": imp, "wins": wins, "losses": n - wins, "p": p})
        return pd.DataFrame(rows)

    domain_stats = group_stats(merged, "domain").sort_values("improvement_pct", ascending=False)
    freq_stats = group_stats(merged, "freq_bucket")
    freq_stats["sort_key"] = freq_stats["group"].map(FREQ_ORDER)
    freq_stats = freq_stats.sort_values("sort_key")

    # Print tables
    print("\n" + "=" * 90)
    print("PER-DOMAIN BREAKDOWN (HD10 vs BL, 5 seeds pooled)")
    print("=" * 90)
    for _, r in domain_stats.iterrows():
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else "ns"
        print(f"  {r['group']:20s}  n={r['n']:3d}  BL={r['geo_bl']:.4f}  HD10={r['geo_hd']:.4f}  "
              f"Δ={r['improvement_pct']:+.2f}%  W/L={r['wins']}/{r['losses']}  p={r['p']:.3e} {sig}")

    print("\n" + "=" * 90)
    print("PER-FREQUENCY BREAKDOWN (HD10 vs BL, 5 seeds pooled)")
    print("=" * 90)
    for _, r in freq_stats.iterrows():
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else "ns"
        print(f"  {r['group']:15s}  n={r['n']:3d}  BL={r['geo_bl']:.4f}  HD10={r['geo_hd']:.4f}  "
              f"Δ={r['improvement_pct']:+.2f}%  W/L={r['wins']}/{r['losses']}  p={r['p']:.3e} {sig}")

    # Also do per-seed breakdown for variance analysis
    print("\n" + "=" * 90)
    print("PER-SEED SUMMARY")
    print("=" * 90)
    for seed in sorted(merged['seed'].unique()):
        sub = merged[merged['seed'] == seed]
        gm_bl = gmean(sub["MASE_bl"])
        gm_hd = gmean(sub["MASE_hd10"])
        wins = int(sub["hd10_wins"].sum())
        n = len(sub)
        p = binomtest(wins, n, 0.5, alternative='greater').pvalue
        print(f"  Seed {seed:2d}: BL={gm_bl:.4f}  HD10={gm_hd:.4f}  Δ={(1-gm_hd/gm_bl)*100:+.2f}%  "
              f"W/L={wins}/{n-wins}  p={p:.3e}")

    # Plotting
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    POSITIVE_COLOR = "#2E7D32"
    NEGATIVE_COLOR = "#C62828"

    # Domain improvement
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(domain_stats))
    vals = domain_stats["improvement_pct"].values
    colors = [POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in vals]
    ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.65, alpha=0.85)
    labels = []
    for _, r in domain_stats.iterrows():
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
        labels.append(f"{r['group']}  ({r['wins']}W/{r['losses']}L){sig}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        offset = 0.3 if v >= 0 else -0.3
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, i, f"{v:+.1f}%", va="center", ha=ha, fontsize=9, fontweight="bold")
    ax.set_xlabel("Geometric Mean MASE Improvement (%)")
    ax.set_title("HD10 vs Baseline: Improvement by Domain\n(5 seeds pooled, m2d, 10K steps)", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f"domain_improvement_m2d.{ext}"))
    plt.close(fig)
    print(f"\nSaved domain_improvement_m2d.pdf/png")

    # Frequency improvement
    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(freq_stats))
    vals = freq_stats["improvement_pct"].values
    colors = [POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in vals]
    ax.bar(x_pos, vals, color=colors, edgecolor="white", width=0.6, alpha=0.85)
    xlabels = []
    for _, r in freq_stats.iterrows():
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
        xlabels.append(f"{r['group']}\n{r['wins']}W/{r['losses']}L{sig}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels, fontsize=8)
    for i, v in enumerate(vals):
        offset = 0.3 if v >= 0 else -0.3
        va = "bottom" if v >= 0 else "top"
        ax.text(i, v + offset, f"{v:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")
    ax.set_ylabel("Geomean MASE Improvement (%)")
    ax.set_title("HD10 vs Baseline: Improvement by Frequency\n(5 seeds pooled, m2d, 10K steps)", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f"frequency_improvement_m2d.{ext}"))
    plt.close(fig)
    print(f"Saved frequency_improvement_m2d.pdf/png")

    # Per-frequency per-seed heatmap
    print("\n" + "=" * 90)
    print("PER-FREQUENCY × PER-SEED WIN RATES")
    print("=" * 90)
    seeds = sorted(merged['seed'].unique())
    freqs = list(freq_stats['group'])
    header = f"{'Freq':>15s} " + " ".join(f"s{s:>3d}" for s in seeds) + " | Pooled"
    print(header)
    for freq in freqs:
        parts = [f"{freq:>15s}"]
        for seed in seeds:
            sub = merged[(merged['freq_bucket'] == freq) & (merged['seed'] == seed)]
            wr = sub['hd10_wins'].sum() / len(sub) * 100 if len(sub) > 0 else 0
            parts.append(f"{wr:5.0f}%")
        sub_pooled = merged[merged['freq_bucket'] == freq]
        wr_pooled = sub_pooled['hd10_wins'].sum() / len(sub_pooled) * 100
        parts.append(f"| {wr_pooled:5.1f}%")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
