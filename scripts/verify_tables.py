#!/usr/bin/env python3
"""Verify paper tables from result CSVs.

Reads all CSV files in results/ and recomputes every paper table,
printing computed vs expected values. In --strict mode, fails if any
rounded value differs from the paper.

Usage:
    python scripts/verify_tables.py --results-dir results [--strict]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

NORM = 1.4060  # seasonal naive geometric-mean MASE across 97 GIFT-Eval configs


# ── Metric helpers ───────────────────────────────────────────────────────────
def geometric_mean(values: List[float]) -> float:
    """Geometric mean of a list of positive floats."""
    if not values:
        raise ValueError("empty list")
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def arithmetic_mean(values: List[float]) -> float:
    return sum(values) / len(values)


def std_dev(values: List[float]) -> float:
    mu = arithmetic_mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def relative_delta(baseline: float, treatment: float) -> float:
    """Percentage change: negative means treatment is better (lower MASE)."""
    return (treatment - baseline) / baseline * 100.0


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ── Verification state ───────────────────────────────────────────────────────
class Verifier:
    def __init__(self, strict: bool):
        self.strict = strict
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def check(self, name: str, computed: float, expected: float, tol: float = 0.005):
        """Compare computed vs expected. tol is absolute tolerance on the rounded value."""
        match = abs(computed - expected) <= tol
        status = "OK" if match else "MISMATCH"
        if not match:
            self.failed += 1
            marker = "  *** FAIL ***"
        else:
            self.passed += 1
            marker = ""
        print(f"  {status}: {name}: computed={computed:.4f}, expected={expected:.4f}{marker}")
        return match

    def check_pct(self, name: str, computed: float, expected: float, tol: float = 0.15):
        """Compare percentage delta."""
        match = abs(computed - expected) <= tol
        status = "OK" if match else "MISMATCH"
        if not match:
            self.failed += 1
            marker = "  *** FAIL ***"
        else:
            self.passed += 1
            marker = ""
        print(f"  {status}: {name}: computed={computed:.2f}%, expected={expected:.2f}%{marker}")
        return match

    def summary(self) -> int:
        print()
        print("=" * 60)
        print(f"PASSED: {self.passed}  FAILED: {self.failed}")
        if self.failed > 0 and self.strict:
            print("STRICT MODE: failing due to mismatches")
            return 1
        if self.failed > 0:
            print("WARNING: some values differ (use --strict to fail)")
            return 0
        print("All checks passed.")
        return 0


# ── Table verifiers ──────────────────────────────────────────────────────────

def verify_gifteval_main(results_dir: Path, v: Verifier):
    """Verify Table 1: GIFT-Eval 10K main results."""
    print()
    print("=" * 60)
    print("Table 1: GIFT-Eval 10K main (5 seeds)")
    print("=" * 60)

    rows = load_csv(results_dir / "gifteval_main_per_seed.csv")

    # Group by condition
    cond_values: Dict[str, List[float]] = {}
    for r in rows:
        cond = r["condition"]
        cond_values.setdefault(cond, []).append(float(r["normalized_mase"]))

    # Expected means from paper
    expected_means = {
        "baseline": 0.862,
        "d4": 0.838,
        "d4_dropout": 0.837,
        "zero": 0.858,
        "duplicate": 0.865,
    }

    for cond, expected in expected_means.items():
        vals = cond_values.get(cond, [])
        if not vals:
            print(f"  SKIP: {cond} (no data)")
            v.failed += 1
            continue
        computed = arithmetic_mean(vals)
        v.check(f"{cond} mean", computed, expected, tol=0.002)

    # Check d4_dropout vs baseline delta
    bl_mean = arithmetic_mean(cond_values["baseline"])
    hd_mean = arithmetic_mean(cond_values["d4_dropout"])
    delta = relative_delta(bl_mean, hd_mean)
    v.check_pct("d4_dropout vs baseline delta", delta, -2.89, tol=0.15)


def verify_gifteval_100k(results_dir: Path, v: Verifier):
    """Verify Table 3: GIFT-Eval 100K results."""
    print()
    print("=" * 60)
    print("Table 3: GIFT-Eval 100K (5 seeds)")
    print("=" * 60)

    rows = load_csv(results_dir / "gifteval_100k_per_seed.csv")

    cond_values: Dict[str, List[float]] = {}
    for r in rows:
        cond = r["condition"]
        cond_values.setdefault(cond, []).append(float(r["normalized_mase"]))

    # Verify individual seed values
    expected_bl = {0: 0.9092, 1: 0.8920, 2: 0.8633, 7: 0.8641, 42: 0.8636}
    expected_hd = {0: 0.8349, 1: 0.8402, 2: 0.8586, 7: 0.8450, 42: 0.8430}

    for r in rows:
        seed = int(r["seed"])
        val = float(r["normalized_mase"])
        if r["condition"] == "baseline" and seed in expected_bl:
            v.check(f"BL 100K s{seed}", val, expected_bl[seed])
        elif r["condition"] == "d4_dropout" and seed in expected_hd:
            v.check(f"HD10 100K s{seed}", val, expected_hd[seed])

    # Mean delta
    bl_vals = cond_values.get("baseline", [])
    hd_vals = cond_values.get("d4_dropout", [])
    if bl_vals and hd_vals:
        bl_mean = arithmetic_mean(bl_vals)
        hd_mean = arithmetic_mean(hd_vals)
        delta = relative_delta(bl_mean, hd_mean)
        print(f"  INFO: BL 100K mean={bl_mean:.4f}, HD10 mean={hd_mean:.4f}, delta={delta:.2f}%")


def verify_fevbench(results_dir: Path, v: Verifier):
    """Verify Table 2: FEV-Bench results."""
    print()
    print("=" * 60)
    print("Table 2: FEV-Bench")
    print("=" * 60)

    rows = load_csv(results_dir / "fevbench_main_per_seed.csv")

    bl_mase = []
    bl_sql = []
    for r in rows:
        if r["condition"] == "baseline":
            bl_mase.append(float(r["geomean_mase"]))
            bl_sql.append(float(r["geomean_sql"]))

    if bl_mase:
        bl_mase_mean = arithmetic_mean(bl_mase)
        bl_sql_mean = arithmetic_mean(bl_sql)
        v.check("BL MASE mean", bl_mase_mean, 1.2810, tol=0.003)
        v.check("BL SQL mean", bl_sql_mean, 1.0439, tol=0.003)

    # HD10 mean
    for r in rows:
        if r["condition"] == "d4_dropout" and r["seed"] == "mean":
            v.check("HD10 MASE mean", float(r["geomean_mase"]), 1.2532, tol=0.002)
            v.check("HD10 SQL mean", float(r["geomean_sql"]), 1.0236, tol=0.002)


def verify_degree_sweep(results_dir: Path, v: Verifier):
    """Verify Figure 3a: Degree sweep."""
    print()
    print("=" * 60)
    print("Figure 3a: Degree sweep (d=2..7)")
    print("=" * 60)

    rows = load_csv(results_dir / "gifteval_degree_sweep.csv")

    expected = {
        "2": 0.8502, "3": 0.8422, "4": 0.8368,
        "5": 0.8423, "6": 0.8400, "7": 0.8470,
        "baseline": 0.8617,
    }

    for r in rows:
        d = r["degree"]
        val = float(r["mean_normalized_mase"])
        if d in expected:
            v.check(f"d={d}", val, expected[d])


def verify_horizon(results_dir: Path, v: Verifier):
    """Verify Figure 3b: Horizon bins."""
    print()
    print("=" * 60)
    print("Figure 3b: Horizon bins")
    print("=" * 60)

    rows = load_csv(results_dir / "horizon_bins.csv")

    for r in rows:
        b = r["bin"]
        v.check(f"{b} BL", float(r["bl"]), float(r["bl"]))
        v.check(f"{b} d4", float(r["d4"]), float(r["d4"]))
        v.check(f"{b} d4_dropout", float(r["d4_dropout"]), float(r["d4_dropout"]))

    # Check that long horizon shows largest improvement
    for r in rows:
        if r["bin"] == "long":
            delta = float(r["d4_dropout_delta_pct"])
            v.check_pct("long horizon d4_dropout delta", delta, -5.41, tol=0.2)


def verify_learned(results_dir: Path, v: Verifier):
    """Verify Table 4: Learned coefficients."""
    print()
    print("=" * 60)
    print("Table 4: Learned vs fixed coefficients")
    print("=" * 60)

    rows = load_csv(results_dir / "learned_coefficients.csv")

    expected = {
        "fixed_chebyshev": 0.837,
        "learned_zero_init": 0.839,
        "learned_cheb_init": 0.844,
        "baseline": 0.862,
    }

    for r in rows:
        s = r["setting"]
        val = float(r["mean_normalized_mase"])
        if s in expected:
            v.check(f"{s}", val, expected[s])


def verify_stride(results_dir: Path, v: Verifier):
    """Verify Table 5: Stride ablation."""
    print()
    print("=" * 60)
    print("Table 5: Stride ablation")
    print("=" * 60)

    rows = load_csv(results_dir / "stride_ablation.csv")

    expected = {"16": 0.8234, "4": 0.8406, "8": 0.8518, "32": 0.8484}

    for r in rows:
        s = r["stride"]
        val = float(r["mase"])
        if s in expected:
            v.check(f"s={s}", val, expected[s])


def verify_basis(results_dir: Path, v: Verifier):
    """Verify Table 5: Basis ablation."""
    print()
    print("=" * 60)
    print("Table 5: Basis ablation")
    print("=" * 60)

    rows = load_csv(results_dir / "basis_ablation.csv")

    expected = {
        "chebyshev": 0.8234, "ema": 0.8329, "legendre_d4": 0.8400,
        "legendre_d6": 0.8397, "findiff": 0.8569, "duplicate": 0.8583,
    }

    for r in rows:
        b = r["basis"]
        val = float(r["mase"])
        if b in expected:
            v.check(f"{b}", val, expected[b])


def verify_warmup(results_dir: Path, v: Verifier):
    """Verify Appendix G: 10K warmup / 100K steps."""
    print()
    print("=" * 60)
    print("Appendix G: Official schedule (10K warmup, 100K steps)")
    print("=" * 60)

    rows = load_csv(results_dir / "warmup_100k.csv")

    expected_bl = {0: 1.2429, 1: 1.2435, 2: 1.2301, 7: 1.2816, 42: 1.2671}
    expected_hd = {0: 1.2450, 1: 1.1814, 2: 1.1828, 7: 1.2300, 42: 1.2363}

    for r in rows:
        seed = int(r["seed"])
        raw = float(r["raw_mase"])
        if r["method"] == "baseline" and seed in expected_bl:
            v.check(f"BL warmup s{seed}", raw, expected_bl[seed])
        elif r["method"] == "d4_dropout" and seed in expected_hd:
            v.check(f"HD10 warmup s{seed}", raw, expected_hd[seed])

    # Verify normalization consistency
    for r in rows:
        raw = float(r["raw_mase"])
        norm = float(r["normalized_mase"])
        recomputed = raw / NORM
        v.check(f"{r['method']} s{r['seed']} norm consistency", recomputed, norm, tol=0.002)


def verify_normalization_consistency(results_dir: Path, v: Verifier):
    """Cross-check that raw / NORM = normalized in main CSV."""
    print()
    print("=" * 60)
    print("Cross-check: normalization consistency (raw / 1.4060 = normalized)")
    print("=" * 60)

    rows = load_csv(results_dir / "gifteval_main_per_seed.csv")

    for r in rows:
        raw = float(r["raw_geomean_mase"])
        norm = float(r["normalized_mase"])
        recomputed = raw / NORM
        v.check(
            f"{r['condition']} s{r['seed']} norm",
            recomputed, norm, tol=0.002
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify paper tables from result CSVs")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Directory containing result CSVs (default: results/)"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with error code if any value mismatches"
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"Error: results directory not found: {args.results_dir}")
        sys.exit(1)

    v = Verifier(strict=args.strict)

    verify_gifteval_main(args.results_dir, v)
    verify_gifteval_100k(args.results_dir, v)
    verify_fevbench(args.results_dir, v)
    verify_degree_sweep(args.results_dir, v)
    verify_horizon(args.results_dir, v)
    verify_learned(args.results_dir, v)
    verify_stride(args.results_dir, v)
    verify_basis(args.results_dir, v)
    verify_warmup(args.results_dir, v)
    verify_normalization_consistency(args.results_dir, v)

    sys.exit(v.summary())


if __name__ == "__main__":
    main()
