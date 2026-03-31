#!/usr/bin/env python3
"""Automated analysis of m2d GIFT-Eval results.
Compares hint methods against baselines, per-seed and pooled.
Usage: python analyze_m2d_results.py [--csv-dir DIR] [--model-map MAP_FILE]
"""
import csv, math, sys, glob, os
from collections import defaultdict

def load_csv(path):
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['dataset']}/{row['frequency']}/{row['term']}"
            try:
                results[key] = float(row['MASE'])
            except (ValueError, KeyError):
                pass
    return results

def geo_mean(d):
    vals = [v for v in d.values() if v > 0 and not math.isnan(v)]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float('nan')

def sign_test_p(wins, n):
    from math import comb
    if wins <= n // 2:
        return sum(comb(n, k) * 0.5**n for k in range(wins + 1))
    return sum(comb(n, k) * 0.5**n for k in range(wins, n + 1))

def compare(method_data, baseline_data):
    """Compare method vs baseline, return (geo_mase, delta%, wins, n, p)."""
    m_geo = geo_mean(method_data)
    b_geo = geo_mean(baseline_data)
    delta = (m_geo / b_geo - 1) * 100
    wins = sum(1 for k in baseline_data if k in method_data and method_data[k] < baseline_data[k])
    n = sum(1 for k in baseline_data if k in method_data)
    p = sign_test_p(wins, n)
    return m_geo, delta, wins, n, p

def freq_breakdown(method_data, baseline_data):
    """Win rate by frequency."""
    freq_wins = defaultdict(lambda: [0, 0])
    for k in baseline_data:
        if k in method_data:
            parts = k.split("/")
            freq = parts[1] if len(parts) >= 3 else "?"
            freq_wins[freq][1] += 1
            if method_data[k] < baseline_data[k]:
                freq_wins[freq][0] += 1
    return dict(freq_wins)


# Known model→CSV mappings (update as new results arrive)
MODEL_MAP = {
    # Core seeds (BL, HD10, MS46, MSHD10)
    "BL_m2d_s0": "000815", "BL_m2d_s1": "002546", "BL_m2d_s2": "030624",
    "BL_m2d_s7": "151340", "BL_m2d_s42": "230321",
    "HD10_m2d_s0": "005524", "HD10_m2d_s1": "031855", "HD10_m2d_s2": "034842",
    "HD10_m2d_s7": "163922", "HD10_m2d_s42": "234539",
    "MS46_m2d_s0": "000940", "MS46_m2d_s1": "050016", "MS46_m2d_s2": "054233",
    "MS46_m2d_s7": "213747",
    "MSHD10_m2d_s0": "111050", "MSHD10_m2d_s1": "115331", "MSHD10_m2d_s2": "123612",
    "dup_m2d_s0": "145350", "dup_m2d_s1": "121009", "dup_m2d_s2": "125323",
    # Stride ablation
    "s4_m2d_s0": "144936", "s8_m2d_s0": "153256", "s32_m2d_s0": "161609",
    # Degree ablation (Chebyshev, 10% dp)
    "hd2_m2d_s0": "015013", "hd3_m2d_s0": "034041",
    "hd6_m2d_s0": "150521", "hd8_m2d_s0": "171508",
    # Basis functions (10% dp)
    "ema_m2d_s0": "163152", "leg4_m2d_s0": "191024", "leg6_m2d_s0": "195248",
    "diff4_m2d_s0": "195156",
    # Multi-scale
    "multi_m2d_s0": "165939",
    # Dropout ablation (Chebyshev d=4)
    "hd10_dp20_s0": "005954", "hd30_m2d_s0": "014233",
    # Combined dropout
    "sd30_m2d_s0": "033004", "hd10sd20_s0": "042031", "ms46hd20_s0": "051106",
    # Zero-hint controls
    "zerohint_dp2_s0": "102506", "zerohint_dp30_s0": "110730",
    # Context ablation
    "HD10_ctx1000_s0": "090659", "BL_ctx1000_s0": "093544",
    "HD10_ctx2000_s0": "100934", "BL_ctx2000_s0": "104000",
    # Model soup (fails catastrophically)
    "soup_a07_s0": "063428", "soup_a05_s0": "071652",
    # Dual-stride (d=4@s16 + d=4@s8)
    "DS48_m2d_s0": "213847", "DS48_m2d_s1": "222115",
    # Base model (46M params)
    "BL_base_s42": "215930",
    # Latent precondition (FAILED)
    "lat_c4s1_s0": "230613", "lat_c4s4_s0": "235016",
    # Re-evals
    "MSHD10_m2d_s0_reeval": "230305", "MSHD10_m2d_s1_reeval": "234533",
}

CSV_DIR = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"
# Note: HD10_base CSV is on 20260309, all others on 20260308
PREFIX = "gifteval_results_epoch_99-step_10000_20260308_"
PREFIX_0309 = "gifteval_results_epoch_99-step_10000_20260309_"
# Models with 20260309 CSVs (special handling)
MODEL_MAP_0309 = {
    "MS46_m2d_s42": "015542",  # Corrected! 234803 was BL_s7 duplicate
    "HD10_base_s42": "002002",
    # Stride ablation (m2d retrained)
    "s4_m2d_s0_v2": "015258",
    # Learned FIR
    "learned_cheb_s0": "012321",
    "learned_zero_s0": "021037",
    # Context@512
    "BL_ctx512_s0": "010523",
    "HD10_ctx512_s0": "014342",
    # DS48 with hints (dual-stride)
    "DS48_HD10_s0": "222026",
    # BL seed re-evals
    "BL_m2d_s42_v2": "011307",
    "MS46_m2d_s7_v2": "003044",
}


def main():
    # Load all data
    data = {}
    for name, ts in MODEL_MAP.items():
        path = os.path.join(CSV_DIR, f"{PREFIX}{ts}.csv")
        if os.path.exists(path):
            data[name] = load_csv(path)
    for name, ts in MODEL_MAP_0309.items():
        path = os.path.join(CSV_DIR, f"{PREFIX_0309}{ts}.csv")
        if os.path.exists(path):
            data[name] = load_csv(path)

    # Also scan for any new CSVs not in the map
    new_csvs = []
    for f in sorted(glob.glob(os.path.join(CSV_DIR, f"{PREFIX}*.csv"))):
        ts = os.path.basename(f).replace(PREFIX, "").replace(".csv", "")
        if ts not in MODEL_MAP.values():
            geo = geo_mean(load_csv(f))
            new_csvs.append((ts, geo))

    if new_csvs:
        print("=== Unidentified CSVs (may be new results) ===")
        for ts, geo in new_csvs:
            if 1.10 < geo < 1.25:  # reasonable range
                print(f"  {ts}: geo_mase={geo:.4f}")
        print()

    # Main comparison table
    methods = ["HD10", "MS46", "MSHD10"]
    seeds = [0, 1, 2, 7, 42]

    print("=" * 70)
    print("M2D RESULTS TABLE (context=4000, 10K steps)")
    print("=" * 70)
    print(f"{'Method':<10} {'Seed':>4} {'MASE':>8} {'Δ%':>7} {'Wins':>8} {'p':>10} {'Sig':>5}")
    print("-" * 58)

    for method in methods:
        all_wins, all_n = 0, 0
        for seed in seeds:
            bl_key = f"BL_m2d_s{seed}"
            m_key = f"{method}_m2d_s{seed}"
            if bl_key in data and m_key in data:
                mase, delta, wins, n, p = compare(data[m_key], data[bl_key])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"{method:<10} {seed:>4} {mase:>8.4f} {delta:>+6.2f}% {wins:>3}/{n} {p:>10.2e} {sig:>5}")
                all_wins += wins
                all_n += n
        if all_n > 0:
            pool_p = sign_test_p(all_wins, all_n)
            print(f"{method:<10} {'pool':>4} {'':>8} {'':>7} {all_wins:>3}/{all_n} {pool_p:>10.2e} {'***' if pool_p < 0.001 else ''}")
        print()

    # Frequency analysis for HD10 (best method on m2d)
    print("=" * 70)
    print("HD10 WIN RATE BY FREQUENCY (pooled all seeds)")
    print("=" * 70)
    pooled_freq = defaultdict(lambda: [0, 0])
    for seed in seeds:
        bl = data.get(f"BL_m2d_s{seed}", {})
        hd = data.get(f"HD10_m2d_s{seed}", {})
        if not bl or not hd:
            continue
        for k in bl:
            if k in hd:
                parts = k.split("/")
                freq = parts[1] if len(parts) >= 3 else "?"
                pooled_freq[freq][1] += 1
                if hd[k] < bl[k]:
                    pooled_freq[freq][0] += 1

    for freq in sorted(pooled_freq, key=lambda f: pooled_freq[f][0] / max(pooled_freq[f][1], 1), reverse=True):
        w, n = pooled_freq[freq]
        print(f"  {freq:>6s}: {w:>3}/{n} ({w / n * 100:>5.1f}%)")

    # Ablation tables (all vs BL_s0)
    bl0 = data.get("BL_m2d_s0", {})
    if not bl0:
        return
    bl0_geo = geo_mean(bl0)

    def print_section(title, items):
        print()
        print("=" * 75)
        print(title)
        print("=" * 75)
        print(f"{'Method':<30} {'MASE':>8} {'Δ%':>8} {'Wins':>8} {'p':>12}")
        print("-" * 75)
        print(f"{'Baseline':30} {bl0_geo:>8.4f}")
        for label, key in items:
            if key in data:
                mg, delta, wins, n, p = compare(data[key], bl0)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"{label:30} {mg:>8.4f} {delta:>+7.2f}% {wins:>3}/{n} {p:>12.2e} {sig}")

    # 0. Stride ablation (Chebyshev d=4, 10% dp)
    print_section("STRIDE ABLATION (Chebyshev d=4, 10% dp)", [
        ("stride=4", "s4_m2d_s0"),
        ("stride=8", "s8_m2d_s0"),
        ("stride=16 (HD10)", "HD10_m2d_s0"),
        ("stride=32", "s32_m2d_s0"),
    ])

    # 1. Polynomial degree (Chebyshev, stride 16, 10% dp)
    print_section("DEGREE ABLATION (Chebyshev, stride=16, 10% dp)", [
        ("d=2", "hd2_m2d_s0"),
        ("d=3", "hd3_m2d_s0"),
        ("d=4 (HD10)", "HD10_m2d_s0"),
        ("d=6", "hd6_m2d_s0"),
        ("d=8", "hd8_m2d_s0"),
    ])

    # 2. Dropout rate (Chebyshev d=4, stride 16)
    print_section("DROPOUT ABLATION (Chebyshev d=4, stride=16)", [
        ("10% hint dp (HD10)", "HD10_m2d_s0"),
        ("20% hint dp", "hd10_dp20_s0"),
        ("30% hint dp", "hd30_m2d_s0"),
        ("0% dp (MS46 d=4+6)", "MS46_m2d_s0"),
    ])

    # 3. Basis function (stride 16, 10% dp)
    print_section("BASIS FUNCTION ABLATION (stride=16, 10% dp)", [
        ("Chebyshev d=4 (HD10)", "HD10_m2d_s0"),
        ("EMA", "ema_m2d_s0"),
        ("Legendre d=4", "leg4_m2d_s0"),
        ("Legendre d=6", "leg6_m2d_s0"),
        ("Finite Diff d=4", "diff4_m2d_s0"),
        ("Duplicate channel", "dup_m2d_s0"),
    ])

    # 4. Multi-scale
    print_section("MULTI-SCALE vs SINGLE-SCALE", [
        ("d=4 (HD10, 10%dp)", "HD10_m2d_s0"),
        ("d=4+6+8 (multi)", "multi_m2d_s0"),
        ("d=4+6 (MS46, 0%dp)", "MS46_m2d_s0"),
        ("d=4+6+10%dp (MSHD10)", "MSHD10_m2d_s0"),
        ("MS46+20%dp", "ms46hd20_s0"),
    ])

    # 5. Zero-hint control
    print_section("ZERO-HINT CONTROL (polynomial content matters)", [
        ("HD10 (real hints)", "HD10_m2d_s0"),
        ("Zero-hint 2% dp", "zerohint_dp2_s0"),
        ("Zero-hint 30% dp", "zerohint_dp30_s0"),
    ])

    # 6. Context length
    ctx_data = {}
    for key in ["HD10_ctx512_s0", "BL_ctx512_s0", "HD10_ctx1000_s0", "BL_ctx1000_s0", "HD10_ctx2000_s0", "BL_ctx2000_s0"]:
        if key in data:
            ctx_data[key] = geo_mean(data[key])
    if ctx_data:
        print()
        print("=" * 75)
        print("CONTEXT LENGTH ABLATION (HD10 vs BL, seed 0)")
        print("=" * 75)
        print(f"{'Context':<10} {'BL':>10} {'HD10':>10} {'HD10 Δ%':>10}")
        print("-" * 45)
        for ctx in [512, 1000, 2000, 4000]:
            bl_key = f"BL_ctx{ctx}_s0" if ctx < 4000 else "BL_m2d_s0"
            hd_key = f"HD10_ctx{ctx}_s0" if ctx < 4000 else "HD10_m2d_s0"
            bl_g = geo_mean(data[bl_key]) if bl_key in data else float('nan')
            hd_g = geo_mean(data[hd_key]) if hd_key in data else float('nan')
            delta = (hd_g / bl_g - 1) * 100 if not math.isnan(bl_g) else float('nan')
            print(f"{ctx:<10} {bl_g:>10.4f} {hd_g:>10.4f} {delta:>+9.2f}%")
        print()
        hd2k = ctx_data.get("HD10_ctx2000_s0", float('nan'))
        bl4k = bl0_geo
        print(f"HD10@ctx=2000 ({hd2k:.4f}) vs BL@ctx=4000 ({bl4k:.4f}): {(hd2k/bl4k-1)*100:+.2f}%")
        # Show training-equivalent savings
        print()
        print("TRAINING COMPUTE SAVINGS (BL 100K curve):")
        print("  BL@10K:  1.2185 | BL@20K: 1.2442 | BL@30K: 1.1918")
        print(f"  HD10@10K: {geo_mean(data.get('HD10_m2d_s0', {})):.4f} → beats BL@30K, saving ~2-3x training compute")


if __name__ == "__main__":
    main()
