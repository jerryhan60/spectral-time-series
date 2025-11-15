#!/usr/bin/env python3
"""
Parse evaluation results from Uni2TS output files and generate CSV summary.

Usage:
    python scripts/parse_eval_results.py <results_directory>
"""

import sys
import re
import csv
from pathlib import Path
from typing import Dict, Optional


def parse_metrics_from_file(output_file: Path) -> Optional[Dict[str, str]]:
    """
    Parse metrics from a single evaluation output file.

    Returns dict with metric names and values, or None if parsing fails.
    """
    try:
        content = output_file.read_text()

        # Look for the metrics table line that starts with "None" followed by numbers
        # Example: "None 69849469551.794 64926659120.223 39546.745 ..."
        pattern = r'^None\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'

        for line in content.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                return {
                    'MSE_mean': match.group(1),
                    'MSE_median': match.group(2),
                    'MAE_median': match.group(3),
                    'MASE_median': match.group(4),
                    'MAPE_median': match.group(5),
                    'sMAPE_median': match.group(6),
                    'MSIS': match.group(7),
                    'RMSE_mean': match.group(8),
                    'NRMSE_mean': match.group(9),
                    'ND_median': match.group(10),
                    'mean_weighted_sum_quantile_loss': match.group(11),
                }

        return None

    except Exception as e:
        print(f"Error parsing {output_file}: {e}", file=sys.stderr)
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_eval_results.py <results_directory>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)

    # Find all output files
    output_files = list(results_dir.glob("*_output.txt"))

    if not output_files:
        print(f"No output files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(output_files)} output files")

    # Parse all files
    results = []
    for output_file in sorted(output_files):
        dataset_name = output_file.stem.replace('_output', '')
        print(f"Parsing {dataset_name}...", end=' ')

        metrics = parse_metrics_from_file(output_file)

        if metrics:
            results.append({
                'dataset': dataset_name,
                **metrics,
                'status': 'success'
            })
            print("✓")
        else:
            results.append({
                'dataset': dataset_name,
                'MSE_mean': '',
                'MSE_median': '',
                'MAE_median': '',
                'MASE_median': '',
                'MAPE_median': '',
                'sMAPE_median': '',
                'MSIS': '',
                'RMSE_mean': '',
                'NRMSE_mean': '',
                'ND_median': '',
                'mean_weighted_sum_quantile_loss': '',
                'status': 'failed'
            })
            print("✗")

    # Write CSV
    csv_file = results_dir / 'evaluation_metrics.csv'

    if results:
        fieldnames = [
            'dataset', 'MSE_mean', 'MSE_median', 'MAE_median', 'MASE_median',
            'MAPE_median', 'sMAPE_median', 'MSIS', 'RMSE_mean', 'NRMSE_mean',
            'ND_median', 'mean_weighted_sum_quantile_loss', 'status'
        ]

        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults written to: {csv_file}")

        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        print(f"\nSummary:")
        print(f"  Total: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
    else:
        print("No results to write")


if __name__ == '__main__':
    main()
