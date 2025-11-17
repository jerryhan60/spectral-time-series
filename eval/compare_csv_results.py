#!/usr/bin/env python3
"""
Compare CSV results from different evaluation runs.

This script loads multiple CSV files from evaluation runs and creates
a comparison table showing performance differences.

Usage:
    python compare_csv_results.py \
        --baseline-orig eval/outputs/eval_results_baseline_*/evaluation_metrics.csv \
        --precond-orig eval/outputs/eval_results_precond_*/evaluation_metrics.csv \
        --baseline-precond eval/outputs/eval_baseline_precond_space_*/evaluation_metrics_baseline_in_precond_space.csv \
        --precond-precond eval/outputs/eval_precond_results_*/evaluation_metrics_precond_space.csv \
        --output comparison_results.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import glob


def load_csv(pattern: str, label: str) -> pd.DataFrame:
    """Load CSV from glob pattern."""
    files = glob.glob(pattern)
    if not files:
        print(f"WARNING: No files found for pattern: {pattern}")
        return None

    # Use most recent file if multiple matches
    files.sort()
    csv_file = files[-1]
    print(f"Loading {label}: {csv_file}")

    df = pd.read_csv(csv_file)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results from different runs"
    )
    parser.add_argument(
        "--baseline-orig",
        required=True,
        help="Path/pattern to baseline model CSV (original space)",
    )
    parser.add_argument(
        "--precond-orig",
        required=True,
        help="Path/pattern to preconditioned model CSV (original space)",
    )
    parser.add_argument(
        "--baseline-precond",
        required=False,
        help="Path/pattern to baseline model CSV (preconditioned space)",
    )
    parser.add_argument(
        "--precond-precond",
        required=False,
        help="Path/pattern to preconditioned model CSV (preconditioned space)",
    )
    parser.add_argument(
        "--output",
        default="comparison_results.csv",
        help="Output CSV file (default: comparison_results.csv)",
    )
    parser.add_argument(
        "--metric",
        default="MAE_median",
        help="Metric to compare (default: MAE_median)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Model Comparison Tool")
    print("=" * 80)
    print()

    # Load CSVs
    df_baseline_orig = load_csv(args.baseline_orig, "Baseline (original space)")
    df_precond_orig = load_csv(args.precond_orig, "Preconditioned (original space)")

    if df_baseline_orig is None or df_precond_orig is None:
        print("ERROR: Could not load required CSV files")
        return

    # Optional: preconditioned space comparisons
    df_baseline_precond = None
    df_precond_precond = None

    if args.baseline_precond:
        df_baseline_precond = load_csv(
            args.baseline_precond, "Baseline (preconditioned space)"
        )

    if args.precond_precond:
        df_precond_precond = load_csv(
            args.precond_precond, "Preconditioned (preconditioned space)"
        )

    print()

    # Merge dataframes
    comparison = df_baseline_orig[["dataset"]].copy()

    # Add original space metrics
    if args.metric in df_baseline_orig.columns:
        comparison[f"baseline_orig_{args.metric}"] = df_baseline_orig[args.metric]
    if args.metric in df_precond_orig.columns:
        comparison[f"precond_orig_{args.metric}"] = df_precond_orig[args.metric]

    # Calculate improvement in original space
    if args.metric in df_baseline_orig.columns and args.metric in df_precond_orig.columns:
        baseline_vals = pd.to_numeric(df_baseline_orig[args.metric], errors='coerce')
        precond_vals = pd.to_numeric(df_precond_orig[args.metric], errors='coerce')
        improvement = ((baseline_vals - precond_vals) / baseline_vals * 100)
        comparison["improvement_orig_%"] = improvement

    # Add preconditioned space metrics if available
    if df_baseline_precond is not None:
        # Match datasets
        df_baseline_precond_matched = df_baseline_precond.set_index("dataset").reindex(
            comparison["dataset"]
        ).reset_index()
        if args.metric in df_baseline_precond_matched.columns:
            comparison[f"baseline_precond_{args.metric}"] = df_baseline_precond_matched[
                args.metric
            ]

    if df_precond_precond is not None:
        df_precond_precond_matched = df_precond_precond.set_index("dataset").reindex(
            comparison["dataset"]
        ).reset_index()
        if args.metric in df_precond_precond_matched.columns:
            comparison[f"precond_precond_{args.metric}"] = df_precond_precond_matched[
                args.metric
            ]

    # Calculate improvement in preconditioned space
    if (
        df_baseline_precond is not None
        and df_precond_precond is not None
        and args.metric in df_baseline_precond.columns
        and args.metric in df_precond_precond.columns
    ):
        baseline_p_vals = pd.to_numeric(
            comparison[f"baseline_precond_{args.metric}"], errors='coerce'
        )
        precond_p_vals = pd.to_numeric(
            comparison[f"precond_precond_{args.metric}"], errors='coerce'
        )
        improvement_p = ((baseline_p_vals - precond_p_vals) / baseline_p_vals * 100)
        comparison["improvement_precond_%"] = improvement_p

    # Save comparison
    comparison.to_csv(args.output, index=False)
    print(f"Comparison saved to: {args.output}")
    print()

    # Display summary
    print("=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print()
    print(comparison.to_string(index=False))
    print()

    # Summary statistics
    print("=" * 80)
    print("Aggregate Statistics")
    print("=" * 80)

    if "improvement_orig_%" in comparison.columns:
        orig_improvements = pd.to_numeric(comparison["improvement_orig_%"], errors='coerce')
        valid_orig = orig_improvements.dropna()
        if len(valid_orig) > 0:
            print(f"\nOriginal Space Improvement:")
            print(f"  Mean:   {valid_orig.mean():+.2f}%")
            print(f"  Median: {valid_orig.median():+.2f}%")
            print(f"  Min:    {valid_orig.min():+.2f}%")
            print(f"  Max:    {valid_orig.max():+.2f}%")
            print(f"  Datasets improved: {(valid_orig > 0).sum()}/{len(valid_orig)}")

    if "improvement_precond_%" in comparison.columns:
        precond_improvements = pd.to_numeric(comparison["improvement_precond_%"], errors='coerce')
        valid_precond = precond_improvements.dropna()
        if len(valid_precond) > 0:
            print(f"\nPreconditioned Space Improvement:")
            print(f"  Mean:   {valid_precond.mean():+.2f}%")
            print(f"  Median: {valid_precond.median():+.2f}%")
            print(f"  Min:    {valid_precond.min():+.2f}%")
            print(f"  Max:    {valid_precond.max():+.2f}%")
            print(f"  Datasets improved: {(valid_precond > 0).sum()}/{len(valid_precond)}")

    print()


if __name__ == "__main__":
    main()
