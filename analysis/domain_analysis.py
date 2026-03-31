"""
Per-domain and per-frequency breakdown of GIFT-Eval results.

Compares baseline (Moirai2 Small, 10K steps) vs multi-scale Chebyshev d=4+d=6
hint preconditioning (ms46, 10K steps).

Outputs:
  - analysis/figures/domain_improvement.pdf
  - analysis/figures/frequency_improvement.pdf
  - analysis/tables/table_domain.tex
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gmean

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
BASELINE_CSV = os.path.join(
    RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260219_121557.csv"
)
MS46_CSV = os.path.join(
    RESULTS_DIR, "gifteval_results_epoch_99-step_10000_20260223_163428.csv"
)
FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
TABLE_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/tables"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
baseline = pd.read_csv(BASELINE_CSV)
ms46 = pd.read_csv(MS46_CSV)

# Merge on the full config (dataset + term) to align rows
baseline["config"] = baseline["dataset"] + "/" + baseline["term"]
ms46["config"] = ms46["dataset"] + "/" + ms46["term"]

merged = baseline[["config", "dataset", "term", "frequency", "MASE"]].merge(
    ms46[["config", "MASE"]],
    on="config",
    suffixes=("_base", "_ms46"),
)

assert len(merged) == 97, f"Expected 97 rows, got {len(merged)}"

# Verify geomean matches expectations
base_gmean = gmean(merged["MASE_base"])
ms46_gmean = gmean(merged["MASE_ms46"])
print(f"Baseline geomean MASE: {base_gmean:.4f}")
print(f"ms46     geomean MASE: {ms46_gmean:.4f}")
print(f"Overall improvement:   {(1 - ms46_gmean / base_gmean) * 100:.2f}%")
print()

# ---------------------------------------------------------------------------
# Domain categorisation
# ---------------------------------------------------------------------------
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

merged["base_dataset"] = merged["dataset"].apply(lambda x: x.split("/")[0])
merged["domain"] = merged["base_dataset"].map(DOMAIN_MAP).fillna("Other")

# ---------------------------------------------------------------------------
# Frequency categorisation
# ---------------------------------------------------------------------------
# Map raw frequency strings to human-readable labels and sort order
FREQ_ORDER = {
    "sub-hourly": 0,
    "hourly": 1,
    "daily": 2,
    "weekly": 3,
    "monthly": 4,
    "quarterly": 5,
    "yearly": 6,
}


def classify_freq(raw_freq: str) -> str:
    """Map pandas-style frequency codes to readable buckets."""
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
    return f  # fallback


merged["freq_bucket"] = merged["frequency"].apply(classify_freq)

# ---------------------------------------------------------------------------
# Helper: compute group-level statistics
# ---------------------------------------------------------------------------


def group_stats(df, group_col):
    """Compute geomean MASE, improvement, and win/loss per group."""
    rows = []
    for grp, sub in df.groupby(group_col):
        n = len(sub)
        geo_base = gmean(sub["MASE_base"])
        geo_ms46 = gmean(sub["MASE_ms46"])
        improvement_pct = (1 - geo_ms46 / geo_base) * 100
        wins = int((sub["MASE_ms46"] < sub["MASE_base"]).sum())
        losses = int((sub["MASE_ms46"] > sub["MASE_base"]).sum())
        ties = n - wins - losses
        rows.append(
            {
                "group": grp,
                "n": n,
                "geo_base": geo_base,
                "geo_ms46": geo_ms46,
                "improvement_pct": improvement_pct,
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
        )
    return pd.DataFrame(rows)


domain_stats = group_stats(merged, "domain").sort_values(
    "improvement_pct", ascending=False
)
freq_stats = group_stats(merged, "freq_bucket")
freq_stats["sort_key"] = freq_stats["group"].map(FREQ_ORDER)
freq_stats = freq_stats.sort_values("sort_key")

# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------
print("=" * 80)
print("PER-DOMAIN BREAKDOWN")
print("=" * 80)
for _, r in domain_stats.iterrows():
    print(
        f"  {r['group']:20s}  n={r['n']:2d}  "
        f"base={r['geo_base']:.4f}  ms46={r['geo_ms46']:.4f}  "
        f"imp={r['improvement_pct']:+.2f}%  "
        f"W/L={r['wins']}/{r['losses']}"
    )

print()
print("=" * 80)
print("PER-FREQUENCY BREAKDOWN")
print("=" * 80)
for _, r in freq_stats.iterrows():
    print(
        f"  {r['group']:15s}  n={r['n']:2d}  "
        f"base={r['geo_base']:.4f}  ms46={r['geo_ms46']:.4f}  "
        f"imp={r['improvement_pct']:+.2f}%  "
        f"W/L={r['wins']}/{r['losses']}"
    )
print()

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

POSITIVE_COLOR = "#2E7D32"  # green  — improvement
NEGATIVE_COLOR = "#C62828"  # red    — regression


# ---------------------------------------------------------------------------
# Figure 1: Domain improvement bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
y_pos = np.arange(len(domain_stats))
vals = domain_stats["improvement_pct"].values
colors = [POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in vals]
bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.65, alpha=0.85)

# Labels on y-axis
labels = [
    f"{r['group']}  ({r['wins']}W/{r['losses']}L, n={r['n']})"
    for _, r in domain_stats.iterrows()
]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()

# Value annotations
for i, v in enumerate(vals):
    offset = 0.3 if v >= 0 else -0.3
    ha = "left" if v >= 0 else "right"
    ax.text(v + offset, i, f"{v:+.1f}%", va="center", ha=ha, fontsize=9, fontweight="bold")

ax.set_xlabel("Geometric Mean MASE Improvement (%)")
ax.set_title("ms46 vs Baseline: Improvement by Domain (10K steps)", fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8)
ax.grid(True, alpha=0.25, axis="x")
ax.set_xlim(
    min(vals.min() - 4, -2),
    max(vals.max() + 4, 2),
)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "domain_improvement.pdf"))
fig.savefig(os.path.join(FIG_DIR, "domain_improvement.png"))
print(f"Saved {os.path.join(FIG_DIR, 'domain_improvement.pdf')}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Frequency improvement bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.5))
x_pos = np.arange(len(freq_stats))
vals = freq_stats["improvement_pct"].values
colors = [POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in vals]
bars = ax.bar(x_pos, vals, color=colors, edgecolor="white", width=0.6, alpha=0.85)

# Labels on x-axis
xlabels = [
    f"{r['group']}\n{r['wins']}W/{r['losses']}L, n={r['n']}"
    for _, r in freq_stats.iterrows()
]
ax.set_xticks(x_pos)
ax.set_xticklabels(xlabels, fontsize=8)

# Value annotations
for i, v in enumerate(vals):
    offset = 0.3 if v >= 0 else -0.3
    va = "bottom" if v >= 0 else "top"
    ax.text(i, v + offset, f"{v:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")

ax.set_ylabel("Geomean MASE Improvement (%)")
ax.set_title("ms46 vs Baseline: Improvement by Frequency (10K steps)", fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(True, alpha=0.25, axis="y")
ax.set_ylim(
    min(vals.min() - 4, -2),
    max(vals.max() + 4, 2),
)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "frequency_improvement.pdf"))
fig.savefig(os.path.join(FIG_DIR, "frequency_improvement.png"))
print(f"Saved {os.path.join(FIG_DIR, 'frequency_improvement.pdf')}")
plt.close(fig)

# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Per-domain and per-frequency improvement of multi-scale Chebyshev hint (ms46) over baseline at 10K steps on GIFT-Eval.}")
lines.append(r"\label{tab:domain_freq}")
lines.append(r"\small")

# Part A: Domain
lines.append(r"\begin{subtable}[t]{0.48\textwidth}")
lines.append(r"\centering")
lines.append(r"\caption{By domain}")
lines.append(r"\begin{tabular}{lrrrrr}")
lines.append(r"\toprule")
lines.append(r"Domain & $n$ & Base & ms46 & Impr.\ (\%) & W/L \\")
lines.append(r"\midrule")
for _, r in domain_stats.iterrows():
    imp_str = f"{r['improvement_pct']:+.1f}"
    bold = r["improvement_pct"] > 0
    if bold:
        imp_cell = r"\textbf{" + imp_str + r"}"
    else:
        imp_cell = imp_str
    lines.append(
        f"{r['group']} & {r['n']} & {r['geo_base']:.3f} & {r['geo_ms46']:.3f} & "
        f"{imp_cell} & {r['wins']}/{r['losses']} \\\\"
    )
lines.append(r"\midrule")
# Overall row
lines.append(
    f"\\textbf{{Overall}} & 97 & {base_gmean:.3f} & {ms46_gmean:.3f} & "
    f"\\textbf{{{(1 - ms46_gmean / base_gmean) * 100:+.1f}}} & "
    f"{int((merged['MASE_ms46'] < merged['MASE_base']).sum())}/"
    f"{int((merged['MASE_ms46'] > merged['MASE_base']).sum())} \\\\"
)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{subtable}")
lines.append(r"\hfill")

# Part B: Frequency
lines.append(r"\begin{subtable}[t]{0.48\textwidth}")
lines.append(r"\centering")
lines.append(r"\caption{By frequency}")
lines.append(r"\begin{tabular}{lrrrrr}")
lines.append(r"\toprule")
lines.append(r"Frequency & $n$ & Base & ms46 & Impr.\ (\%) & W/L \\")
lines.append(r"\midrule")
for _, r in freq_stats.iterrows():
    imp_str = f"{r['improvement_pct']:+.1f}"
    bold = r["improvement_pct"] > 0
    if bold:
        imp_cell = r"\textbf{" + imp_str + r"}"
    else:
        imp_cell = imp_str
    lines.append(
        f"{r['group']} & {r['n']} & {r['geo_base']:.3f} & {r['geo_ms46']:.3f} & "
        f"{imp_cell} & {r['wins']}/{r['losses']} \\\\"
    )
lines.append(r"\midrule")
lines.append(
    f"\\textbf{{Overall}} & 97 & {base_gmean:.3f} & {ms46_gmean:.3f} & "
    f"\\textbf{{{(1 - ms46_gmean / base_gmean) * 100:+.1f}}} & "
    f"{int((merged['MASE_ms46'] < merged['MASE_base']).sum())}/"
    f"{int((merged['MASE_ms46'] > merged['MASE_base']).sum())} \\\\"
)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{subtable}")
lines.append(r"\end{table}")

latex_str = "\n".join(lines)
tex_path = os.path.join(TABLE_DIR, "table_domain.tex")
with open(tex_path, "w") as f:
    f.write(latex_str)
print(f"Saved {tex_path}")

# ---------------------------------------------------------------------------
# Per-dataset detail (for reference)
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("PER-DATASET DETAIL (sorted by improvement)")
print("=" * 80)
merged["improvement_pct"] = (1 - merged["MASE_ms46"] / merged["MASE_base"]) * 100
detail = merged.sort_values("improvement_pct", ascending=False)
for _, r in detail.iterrows():
    marker = "+" if r["improvement_pct"] > 0 else " "
    print(
        f"  {r['config']:50s}  {r['domain']:18s}  {r['freq_bucket']:12s}  "
        f"base={r['MASE_base']:.4f}  ms46={r['MASE_ms46']:.4f}  "
        f"{marker}{r['improvement_pct']:.1f}%"
    )
