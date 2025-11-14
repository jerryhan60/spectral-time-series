#!/usr/bin/env python3
"""
Analyze dataset scales to understand why some have extremely high MSE.
Run this to check if certain datasets have outlier series with very large values.
"""

import sys
from pathlib import Path
import numpy as np

# Add uni2ts to path
sys.path.insert(0, str(Path(__file__).parent / "uni2ts" / "src"))

from uni2ts.eval_util.data import get_gluonts_test_dataset


def analyze_dataset(dataset_name: str):
    """Analyze scale and statistics of a dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {dataset_name}")
    print(f"{'='*60}")

    try:
        # Load dataset
        test_data, metadata = get_gluonts_test_dataset(
            dataset_name=dataset_name,
            prediction_length=None,
            mode='S',
            use_lotsa_cache=True
        )

        print(f"Prediction length: {metadata.prediction_length}")
        print(f"Frequency: {metadata.freq}")
        print(f"Target dim: {metadata.target_dim}")

        # Collect statistics
        all_values = []
        series_stats = []

        for i, entry in enumerate(test_data):
            target = entry['target']
            values = target.flatten() if hasattr(target, 'flatten') else np.array(target).flatten()

            # Remove NaN/Inf
            values = values[np.isfinite(values)]

            if len(values) > 0:
                all_values.extend(values.tolist())
                series_stats.append({
                    'series_idx': i,
                    'length': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'abs_mean': np.mean(np.abs(values))
                })

            # Limit to first 1000 series for speed
            if i >= 1000:
                break

        # Overall statistics
        all_values = np.array(all_values)
        print(f"\nNumber of series analyzed: {len(series_stats)}")
        print(f"Total values: {len(all_values):,}")

        print(f"\nOverall value statistics:")
        print(f"  Mean: {np.mean(all_values):.2f}")
        print(f"  Std:  {np.std(all_values):.2f}")
        print(f"  Min:  {np.min(all_values):.2f}")
        print(f"  Max:  {np.max(all_values):.2f}")
        print(f"  Median: {np.median(all_values):.2f}")

        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9]
        print(f"\nValue percentiles:")
        for p in percentiles:
            val = np.percentile(np.abs(all_values), p)
            print(f"  {p:5.1f}%: {val:15.2f}")

        # Series-level statistics
        series_means = [s['abs_mean'] for s in series_stats]
        series_maxes = [s['max'] for s in series_stats]

        print(f"\nSeries-level statistics (mean absolute values):")
        print(f"  Min:    {np.min(series_means):.2f}")
        print(f"  Median: {np.median(series_means):.2f}")
        print(f"  Mean:   {np.mean(series_means):.2f}")
        print(f"  Max:    {np.max(series_means):.2f}")
        print(f"  Std:    {np.std(series_means):.2f}")

        # Find outlier series (top 5 by mean value)
        series_stats_sorted = sorted(series_stats, key=lambda x: x['abs_mean'], reverse=True)
        print(f"\nTop 5 series by mean absolute value:")
        for i, s in enumerate(series_stats_sorted[:5]):
            print(f"  {i+1}. Series {s['series_idx']}: mean={s['abs_mean']:.2f}, max={s['max']:.2f}, std={s['std']:.2f}")

        # Check for extreme outliers
        q99 = np.percentile(series_means, 99)
        extreme = [s for s in series_stats if s['abs_mean'] > q99]
        print(f"\nNumber of series above 99th percentile: {len(extreme)}")

        # Estimate MSE contribution from scale
        print(f"\nEstimated MSE impact:")
        print(f"  If predictions have ~30% relative error:")
        mean_scale = np.mean(series_means)
        estimated_mse = (0.3 * mean_scale) ** 2
        print(f"    Estimated MSE per value: {estimated_mse:.2e}")
        print(f"    For {metadata.prediction_length} predictions: {estimated_mse * metadata.prediction_length:.2e}")

        return {
            'dataset_name': dataset_name,
            'n_series': len(series_stats),
            'mean_value': np.mean(all_values),
            'median_series_scale': np.median(series_means),
            'max_series_scale': np.max(series_means),
            'estimated_mse': estimated_mse * metadata.prediction_length
        }

    except Exception as e:
        print(f"✗ Error analyzing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("Dataset Scale Analysis")
    print("="*60)
    print("This analyzes why some datasets have very high MSE values.")
    print()

    # Datasets with high MSE
    high_mse_datasets = [
        'tourism_monthly',
        'm1_monthly',
        'tourism_yearly',
        'm1_yearly',
    ]

    # Datasets with normal MSE (for comparison)
    normal_mse_datasets = [
        'm4_monthly',
        'monash_m3_monthly',
    ]

    print("="*60)
    print("HIGH MSE DATASETS")
    print("="*60)

    high_results = []
    for ds in high_mse_datasets:
        result = analyze_dataset(ds)
        if result:
            high_results.append(result)

    print("\n" + "="*60)
    print("NORMAL MSE DATASETS (for comparison)")
    print("="*60)

    normal_results = []
    for ds in normal_mse_datasets:
        result = analyze_dataset(ds)
        if result:
            normal_results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nHigh MSE datasets:")
    for r in high_results:
        print(f"  {r['dataset_name']:25s} - Median scale: {r['median_series_scale']:12.2f}, Est. MSE: {r['estimated_mse']:.2e}")

    print("\nNormal MSE datasets:")
    for r in normal_results:
        print(f"  {r['dataset_name']:25s} - Median scale: {r['median_series_scale']:12.2f}, Est. MSE: {r['estimated_mse']:.2e}")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("MSE is an absolute metric and is highly sensitive to data scale.")
    print("Datasets with larger values naturally have higher MSE.")
    print()
    print("Recommendation: Focus on scale-independent metrics:")
    print("  - MASE (Mean Absolute Scaled Error)")
    print("  - MAPE (Mean Absolute Percentage Error)")
    print("  - sMAPE (Symmetric Mean Absolute Percentage Error)")
    print("  - ND (Normalized Deviation)")
    print("  - NRMSE (Normalized RMSE)")


if __name__ == "__main__":
    main()
