#!/usr/bin/env python3
"""Analyze inference-time preconditioning ensemble results."""

import sys
import pandas as pd
import numpy as np
from scipy.stats import gmean


def analyze(csv_path: str):
    df = pd.read_csv(csv_path)

    # Filter to successful evaluations
    if "error" in df.columns:
        success = df[df["error"].isna()].copy()
    else:
        success = df.copy()

    print(f"Loaded {len(success)} successful evaluations from {csv_path}")

    mase_cols = sorted([c for c in success.columns if c.endswith("_MASE")])
    if not mase_cols:
        print("No MASE columns found.")
        return

    # === Geometric Mean MASE per method ===
    raw_valid = success["raw_MASE"].dropna()
    raw_valid = raw_valid[(raw_valid > 0) & (raw_valid < 100)]
    raw_gm = gmean(raw_valid) if len(raw_valid) > 0 else float("inf")

    print(f"\n{'='*70}")
    print("GEOMETRIC MEAN MASE (lower is better)")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'Geo Mean':>10} {'vs raw':>10} {'<1.0':>8}")
    print(f"{'-'*70}")

    for col in mase_cols:
        valid = success[col].dropna()
        valid = valid[(valid > 0) & (valid < 100)]
        if len(valid) == 0:
            continue
        gm = gmean(valid)
        vs_raw = (gm / raw_gm - 1) * 100 if raw_gm < float("inf") else 0
        below_1 = (valid < 1.0).sum()
        marker = " ***" if vs_raw < -1 else ""
        print(f"{col:<35} {gm:>10.4f} {vs_raw:>+9.2f}% {below_1:>3}/{len(valid)}{marker}")

    # === Win Rate vs Raw ===
    print(f"\n{'='*70}")
    print("WIN RATE vs RAW (how often does method beat raw per dataset)")
    print(f"{'='*70}")
    for col in mase_cols:
        if col == "raw_MASE":
            continue
        valid = success[["raw_MASE", col]].dropna()
        if len(valid) == 0:
            continue
        wins = (valid[col] < valid["raw_MASE"]).sum()
        ties = (valid[col] == valid["raw_MASE"]).sum()
        losses = len(valid) - wins - ties
        print(f"  {col:<35}: {wins}W/{losses}L/{ties}T out of {len(valid)}")

    # === Per-frequency analysis ===
    if "dataset" in success.columns:
        print(f"\n{'='*70}")
        print("RESULTS BY FREQUENCY CATEGORY")
        print(f"{'='*70}")

        def categorize(name):
            if any(f"/{f}" in str(name) for f in ["5T", "10T", "15T"]):
                return "sub-hourly"
            elif "/H" in str(name):
                return "hourly"
            elif "/D" in str(name):
                return "daily"
            elif any(f"/{f}" in str(name) for f in ["W", "M", "Q"]):
                return "low-freq"
            else:
                return "other"

        success["freq_cat"] = success["dataset"].apply(categorize)

        for cat in ["sub-hourly", "hourly", "daily", "low-freq", "other"]:
            subset = success[success["freq_cat"] == cat]
            if len(subset) == 0:
                continue
            print(f"\n  {cat.upper()} ({len(subset)} configs):")
            for col in mase_cols:
                valid = subset[col].dropna()
                valid = valid[(valid > 0) & (valid < 100)]
                if len(valid) == 0:
                    continue
                gm = gmean(valid)
                raw_sub = subset["raw_MASE"].dropna()
                raw_sub = raw_sub[(raw_sub > 0) & (raw_sub < 100)]
                raw_gm_sub = gmean(raw_sub) if len(raw_sub) > 0 else float("inf")
                vs = (gm / raw_gm_sub - 1) * 100 if raw_gm_sub < float("inf") else 0
                print(f"    {col:<30}: {gm:.4f} ({vs:+.2f}%)")

    # === Per-term analysis ===
    if "term" in success.columns:
        print(f"\n{'='*70}")
        print("RESULTS BY HORIZON TERM")
        print(f"{'='*70}")
        for term in ["short", "medium", "long"]:
            subset = success[success["term"] == term]
            if len(subset) == 0:
                continue
            print(f"\n  {term.upper()} ({len(subset)} configs):")
            for col in ["raw_MASE"] + [c for c in mase_cols if "uniform" in c or "invvar" in c]:
                valid = subset[col].dropna()
                valid = valid[(valid > 0) & (valid < 100)]
                if len(valid) == 0:
                    continue
                print(f"    {col:<30}: {gmean(valid):.4f}")

    # === Best individual degree per strategy ===
    print(f"\n{'='*70}")
    print("BEST INDIVIDUAL DEGREE PER STRATEGY")
    print(f"{'='*70}")
    for strategy in ["A1", "A2"]:
        degree_cols = [c for c in mase_cols if c.startswith(f"{strategy}_d") and c.endswith("_MASE")]
        if not degree_cols:
            continue
        best_col = None
        best_gm = float("inf")
        for col in degree_cols:
            valid = success[col].dropna()
            valid = valid[(valid > 0) & (valid < 100)]
            if len(valid) > 0:
                gm = gmean(valid)
                if gm < best_gm:
                    best_gm = gm
                    best_col = col
        if best_col:
            vs = (best_gm / raw_gm - 1) * 100 if raw_gm < float("inf") else 0
            print(f"  {strategy}: {best_col} = {best_gm:.4f} ({vs:+.2f}% vs raw)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_inference_precond.py <results.csv>")
        sys.exit(1)
    analyze(sys.argv[1])
