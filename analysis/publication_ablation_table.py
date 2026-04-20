#!/usr/bin/env python3
"""
Publication-quality ablation table for polynomial hint preconditioning paper.

Produces Table 1 (main results), Table 2 (ablation study), Table 3 (polynomial family comparison),
and statistical significance tests via paired bootstrap.

Usage:
    python analysis/publication_ablation_table.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RESULTS_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"


def load_gift_eval_results(csv_path):
    """Load a GIFT-Eval results CSV and compute geometric mean MASE."""
    df = pd.read_csv(csv_path)
    # MASE column might be 'MASE[0.5]' or 'eval_metrics/MASE[0.5]'
    mase_col = [c for c in df.columns if 'MASE' in c and '0.5' in c]
    if not mase_col:
        mase_col = [c for c in df.columns if 'MASE' in c]
    if not mase_col:
        return None, None
    mase_col = mase_col[0]
    mase_values = df[mase_col].dropna()
    mase_values = mase_values[mase_values > 0]  # Filter invalid
    geo_mean = np.exp(np.mean(np.log(mase_values)))
    return geo_mean, df


def paired_bootstrap_test(mase_a, mase_b, n_boot=10000, seed=42):
    """Paired bootstrap test: P(geo_mean(A) < geo_mean(B))."""
    rng = np.random.RandomState(seed)
    log_a = np.log(mase_a)
    log_b = np.log(mase_b)
    log_diff = log_a - log_b  # negative = A is better

    boot_diffs = []
    n = len(log_diff)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_diffs.append(np.mean(log_diff[idx]))

    boot_diffs = np.array(boot_diffs)
    p_a_better = np.mean(boot_diffs < 0)
    ratio = np.exp(np.mean(log_diff))
    ci_low = np.exp(np.percentile(boot_diffs, 2.5))
    ci_high = np.exp(np.percentile(boot_diffs, 97.5))

    return ratio, ci_low, ci_high, p_a_better


def find_latest_result(pattern):
    """Find the most recent results CSV matching a pattern."""
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    print("=" * 80)
    print("PUBLICATION ABLATION TABLE — Polynomial Hint Preconditioning")
    print("=" * 80)

    # ========== Known results (hardcoded from completed experiments) ==========
    known_results = {
        # Main results
        "Baseline (no hints)": 1.2421,
        "Hint d=4 (single-scale)": 1.1944,
        "Hint d=6 (single-scale)": 1.1836,
        "Multi-scale d=4+d=6 (best)": 1.1675,
        "Multi-scale d=4+d=6 + 10% drop": 1.1817,
        "Hint d=4 + 10% drop": 1.1802,

        # Polynomial families (all d=6, s=16)
        "Chebyshev d=6": 1.1836,
        "L2-optimized d=6": 1.1784,
        "Lyapunov d=6": 1.1985,
        "Legendre d=6": 1.2099,

        # Legendre multi-scale (NEW)
        "Legendre ms d=4+d=6": 1.2356,
        "Legendre d=5": 1.2237,
        "Legendre ms d=5+d=6": 1.2627,
        "Cross (Cheb d=4 + Leg d=6)": 1.2379,

        # 100K results
        "Baseline 100K (10K warmup)": 1.2911,
        "hd10 100K (1K warmup)": 1.1918,
        "hd10 100K (10K warmup)": 1.2579,
        "ms46 100K (10K warmup)": 1.2214,

        # Other ablations
        "Hint ramp (0→1 over 3K)": 1.2159,
        "Variable prefix [0.15,0.45]": 1.2896,
        "ms46 + lr=5e-4": 1.2687,
        "ms46 + lr=2e-3": 1.2241,
        "Learnable coeffs (d=5)": 1.2025,
    }

    baseline = known_results["Baseline (no hints)"]

    # ========== Table 1: Main Results ==========
    print("\n" + "=" * 80)
    print("TABLE 1: Main Results (10K steps, GIFT-Eval 97 configs)")
    print("=" * 80)
    print(f"{'Method':<40} {'MASE':>8} {'vs Base':>10} {'Params':>10}")
    print("-" * 70)

    main_results = [
        ("Baseline (no hints)", 1.2421, "11.39M"),
        ("Hint d=4 (single-scale)", 1.1944, "11.40M"),
        ("Hint d=6 (single-scale)", 1.1836, "11.40M"),
        ("Hint d=4 + 10% dropout", 1.1802, "11.40M"),
        ("Multi-scale d=4+d=6", 1.1675, "11.40M"),
        ("Multi-scale d=4+d=6 + 10% drop", 1.1817, "11.40M"),
    ]
    for name, mase, params in main_results:
        delta = (mase - baseline) / baseline * 100
        delta_str = f"{delta:+.2f}%" if name != main_results[0][0] else "—"
        print(f"{name:<40} {mase:>8.4f} {delta_str:>10} {params:>10}")

    # ========== Table 2: Ablation Study ==========
    print("\n" + "=" * 80)
    print("TABLE 2: Ablation Study (10K steps)")
    print("What drives the improvement?")
    print("=" * 80)
    print(f"{'Ablation':<45} {'MASE':>8} {'vs Base':>10} {'Status':>12}")
    print("-" * 77)

    ablations = [
        ("Multi-scale d=4+d=6 (Chebyshev)", 1.1675, "Complete"),
        ("Zero hint (ms46 arch, hints=0)", None, "PENDING"),
        ("Random hint (ms46 arch, hints=randn)", None, "PENDING"),
        ("Duplicate input (both channels = target)", None, "PENDING"),
        ("Learned 4-tap (Cheb d=4 init)", None, "PENDING"),
        ("Learned 16-tap (Cheb d=16 init)", None, "PENDING"),
        ("Baseline (no hints, no extra channels)", 1.2421, "Complete"),
    ]

    # Try to find eval results for ablations
    ablation_csvs = {
        "Duplicate input (both channels = target)": "abl_dup",
        "Learned 4-tap (Cheb d=4 init)": "abl_lc4",
        "Learned 16-tap (Cheb d=16 init)": "abl_lc16",
        "Zero hint (ms46 arch, hints=0)": "zero_hint",
        "Random hint (ms46 arch, hints=randn)": "random_hint",
    }

    for name, mase, status in ablations:
        if mase is None and name in ablation_csvs:
            # Try to find results
            pattern = f"gifteval_results_*{ablation_csvs[name]}*.csv"
            csv = find_latest_result(pattern)
            if csv:
                mase, _ = load_gift_eval_results(csv)
                status = "Complete"

        if mase is not None:
            delta = (mase - baseline) / baseline * 100
            delta_str = f"{delta:+.2f}%" if "Baseline" not in name else "—"
            print(f"{name:<45} {mase:>8.4f} {delta_str:>10} {status:>12}")
        else:
            print(f"{name:<45} {'—':>8} {'—':>10} {status:>12}")

    print()
    print("Interpretation:")
    print("  If zero/random/duplicate ≈ baseline: improvement from polynomial structure")
    print("  If learned conv ≈ fixed Chebyshev: structure doesn't matter, any filter works")
    print("  If fixed Chebyshev >> all: polynomial inductive bias is key")

    # ========== Table 3: Polynomial Family Comparison ==========
    print("\n" + "=" * 80)
    print("TABLE 3: Polynomial Family Comparison")
    print("=" * 80)

    print("\n--- Single-Scale (d=6, stride=16, 10K steps) ---")
    print(f"{'Family':<25} {'max|c|':>8} {'MASE':>8} {'vs Base':>10}")
    print("-" * 53)
    families_d6 = [
        ("L2-optimized", "~0.28", 1.1784),
        ("Chebyshev", "1.50", 1.1836),
        ("Lyapunov", "~0.23", 1.1985),
        ("Legendre", "1.36", 1.2099),
    ]
    for name, maxc, mase in families_d6:
        delta = (mase - baseline) / baseline * 100
        print(f"{name:<25} {maxc:>8} {mase:>8.4f} {delta:>+9.2f}%")

    print("\n--- Multi-Scale (d=4+d=6, stride=16, 10K steps) ---")
    print(f"{'Family':<25} {'MASE':>8} {'vs Base':>10} {'vs Cheb ms46':>14}")
    print("-" * 60)
    ms_families = [
        ("Chebyshev", 1.1675),
        ("Legendre", 1.2356),
        ("Cross (Cheb+Leg)", 1.2379),
    ]
    cheb_ms = 1.1675
    for name, mase in ms_families:
        delta_base = (mase - baseline) / baseline * 100
        delta_cheb = (mase - cheb_ms) / cheb_ms * 100
        print(f"{name:<25} {mase:>8.4f} {delta_base:>+9.2f}% {delta_cheb:>+13.2f}%")

    # ========== Table 4: Degree Sweep ==========
    print("\n" + "=" * 80)
    print("TABLE 4: Chebyshev Degree Sweep (single-scale, stride=16)")
    print("=" * 80)
    print(f"{'Degree':>6} {'MASE':>8} {'vs Base':>10}")
    print("-" * 26)
    degrees = [
        (2, 1.2157), (3, 1.2040), (4, 1.1944), (5, 1.2084),
        (6, 1.1836), (7, 1.2027), (8, 1.2216),
    ]
    for d, mase in degrees:
        delta = (mase - baseline) / baseline * 100
        marker = " ← best single" if d == 6 else ""
        print(f"{d:>6} {mase:>8.4f} {delta:>+9.2f}%{marker}")

    # ========== Table 4b: Stride Sweep ==========
    print("\n" + "=" * 80)
    print("TABLE 4b: Stride Sweep (Chebyshev d=6, hint mode)")
    print("Patch-aligned stride (s=16=patch_size) is critical")
    print("=" * 80)
    print(f"{'Stride':>8} {'Relation':>25} {'MASE':>8} {'vs Base':>10}")
    print("-" * 55)
    strides = [
        (1, "sub-patch (dense)", None),
        (4, "sub-patch (¼ patch)", 1.2346),
        (8, "half-patch", 1.2204),
        (16, "patch-aligned (best)", 1.1836),
        (32, "supra-patch (2× patch)", None),
    ]
    # Try to find stride=1 and stride=32 results
    s1_csv = find_latest_result("gifteval_results_*hint_d6_s1*.csv")
    s32_csv = find_latest_result("gifteval_results_*hint_d6_s32*.csv")
    if s1_csv:
        strides[0] = (1, "sub-patch (dense)", load_gift_eval_results(s1_csv)[0])
    if s32_csv:
        strides[4] = (32, "supra-patch (2× patch)", load_gift_eval_results(s32_csv)[0])

    for s, desc, mase in strides:
        if mase:
            delta = (mase - baseline) / baseline * 100
            marker = " ← best" if s == 16 else ""
            print(f"{s:>8} {desc:>25} {mase:>8.4f} {delta:>+9.2f}%{marker}")
        else:
            print(f"{s:>8} {desc:>25} {'—':>8} {'PENDING':>10}")

    # ========== Table 5: 100K Results ==========
    print("\n" + "=" * 80)
    print("TABLE 5: Long Training (100K steps)")
    print("=" * 80)
    baseline_100k = 1.2911
    print(f"{'Method':<45} {'MASE':>8} {'vs 100K BL':>12} {'vs 10K BL':>12}")
    print("-" * 80)
    long_results = [
        ("Baseline 100K (10K warmup)", 1.2911),
        ("hd10 100K (1K warmup, best)", 1.1918),
        ("hd10 100K (10K warmup)", 1.2579),
        ("ms46 100K + 10% drop (10K warmup)", 1.2214),
    ]
    for name, mase in long_results:
        d100k = (mase - baseline_100k) / baseline_100k * 100
        d10k = (mase - baseline) / baseline * 100
        d100k_str = f"{d100k:+.2f}%" if "Baseline 100K" not in name else "—"
        print(f"{name:<45} {mase:>8.4f} {d100k_str:>12} {d10k:>+11.2f}%")

    # ========== Table 6: Per-Horizon Analysis ==========
    print("\n" + "=" * 80)
    print("TABLE 6: Per-Horizon Analysis (ms d=4+d=6 vs Baseline, 10K)")
    print("=" * 80)
    print(f"{'Horizon':<10} {'#Configs':>8} {'Baseline':>10} {'ms46':>10} {'Change':>10} {'Win Rate':>10}")
    print("-" * 60)
    horizons = [
        ("short", 55, 1.1865, 1.1574, 37, 55),
        ("medium", 21, 1.2596, 1.1525, 18, 21),
        ("long", 21, 1.3810, 1.2096, 19, 21),
    ]
    for name, n, bl, ms, wins, total in horizons:
        change = (ms - bl) / bl * 100
        print(f"{name:<10} {n:>8} {bl:>10.4f} {ms:>10.4f} {change:>+9.2f}% {wins}/{total:>7}")

    # ========== Table 7: LR Sweep ==========
    print("\n" + "=" * 80)
    print("TABLE 7: Learning Rate Sweep (10K steps)")
    print("=" * 80)
    print(f"{'LR':>8} {'ms46 MASE':>10} {'Baseline MASE':>14} {'Status':>10}")
    print("-" * 45)
    lr_sweep = [
        ("5e-4", 1.2687, None, "PENDING"),
        ("1e-3", 1.1675, 1.2421, "Complete"),
        ("2e-3", 1.2241, None, "PENDING"),
    ]
    for lr, ms_mase, bl_mase, status in lr_sweep:
        bl_str = f"{bl_mase:.4f}" if bl_mase else "—"
        print(f"{lr:>8} {ms_mase:>10.4f} {bl_str:>14} {status:>10}")

    print()
    print("Per Hazan: 'the usual practice is to do a sweep over learning rates")
    print("and pick the best one, for all methods.' Baseline LR sweep pending.")

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Best 10K model:  Multi-scale Cheb d=4+d=6    MASE {1.1675:.4f} ({(1.1675-baseline)/baseline*100:+.2f}% vs baseline)")
    print(f"  Best 100K model: hd10 (d=4+10%drop, 1K warm) MASE {1.1918:.4f} ({(1.1918-baseline)/baseline*100:+.2f}% vs 10K baseline)")
    print(f"  Official M2.0:   10x more training data       MASE {1.0236:.4f}")
    print()
    print("  Key insight: Polynomial preconditioning hints improve forecasting")
    print("  by 4-8% with <0.1% parameter overhead and zero inference cost.")
    print()

    # ========== Pending Results ==========
    print("PENDING EXPERIMENTS:")
    print("  - Zero/Random/Duplicate hint ablations (confirm polynomial structure matters)")
    print("  - Learned 4-tap/16-tap convolution ablations")
    print("  - Baseline LR sweep (lr=5e-4, lr=2e-3)")
    print("  - bl1k_100k (baseline + 1K warmup at 100K → warmup confound)")
    print("  - Legendre ms46 100K (resuming from 60K)")


if __name__ == "__main__":
    main()
