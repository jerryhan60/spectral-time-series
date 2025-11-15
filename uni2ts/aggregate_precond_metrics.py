#!/usr/bin/env python3
"""
Script to aggregate individual metrics_precond_space.csv files into a summary CSV.
"""

import pandas as pd
from pathlib import Path
import re

# Define the results directory
results_dir = Path("/scratch/gpfs/EHAZAN/jh1161/uni2ts/eval_precond_results_last_20251108_130527")
output_csv = results_dir / "evaluation_metrics_precond_space.csv"

# Find all individual metrics CSV files
metrics_files = list(Path("/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/eval_precond_space").rglob("metrics_precond_space.csv"))

print(f"Found {len(metrics_files)} metrics files")

# Read the existing summary CSV to get the dataset list and order
summary_df = pd.read_csv(output_csv)
datasets = summary_df['dataset'].tolist()

print(f"Found {len(datasets)} datasets in summary CSV")

# Create a mapping from dataset name variations to canonical names
dataset_name_mapping = {
    'australian_electricity_demand': 'Aus_Elec_Demand',
    'bitcoin_with_missing': 'Bitcoin',
    'car_parts_with_missing': 'Carparts',
    'cif_2016': 'CIF_2016',
    'cif_2016_12': 'CIF_2016',  # Alternative naming
    'covid_deaths': 'COVID_Deaths',
    'fred_md': 'FRED_MD',
    'hospital': 'Hospital',
    'kdd_cup_2018_with_missing': 'KDD_Cup',
    'm1_monthly': 'M1_Monthly',
    'm3_monthly': 'M3_Monthly',
    'monash_m3_monthly': 'M3_Monthly',  # Alternative naming
    'm3_other': 'M3_Other',
    'monash_m3_other': 'M3_Other',  # Alternative naming
    'm4_daily': 'M4_Daily',
    'm4_hourly': 'M4_Hourly',
    'm4_monthly': 'M4_Monthly',
    'm4_weekly': 'M4_Weekly',
    'nn5_daily_with_missing': 'NN5_Daily',
    'nn5_weekly': 'NN5_Weekly',
    'pedestrian_counts': 'Pedestrian_Counts',
    'rideshare_with_missing': 'Rideshare',
    'saugeen_river_flow': 'Saugeen_River_Flow',
    'saugeenday': 'Saugeen_River_Flow',  # Alternative naming
    'sunspot_with_missing': 'Sunspot',
    'temperature_rain': 'Temperature_Rain',
    'temperature_rain_with_missing': 'Temperature_Rain',  # Alternative naming
    'tourism_monthly': 'Tourism_Monthly',
    'tourism_quarterly': 'Tourism_Quarterly',
    'traffic_hourly': 'Traffic_Hourly',
    'traffic_weekly': 'Traffic_Weekly',
    'us_births': 'US_Births',
    'vehicle_trips_with_missing': 'Vehicle_Trips',
    'weather': 'Weather',
}

# Dictionary to store metrics for each dataset
dataset_metrics = {}

# Process each metrics file
for metrics_file in metrics_files:
    # Extract dataset name from path
    # Path pattern: .../monash_cached/{dataset_name}/S/...
    path_parts = metrics_file.parts
    try:
        monash_idx = path_parts.index('monash_cached')
        dataset_name_raw = path_parts[monash_idx + 1]
    except (ValueError, IndexError):
        print(f"Could not extract dataset name from {metrics_file}")
        continue

    # Map to canonical name
    canonical_name = dataset_name_mapping.get(dataset_name_raw, dataset_name_raw)

    # Read the metrics
    try:
        metrics = pd.read_csv(metrics_file)
        if len(metrics) > 0:
            # Get the first row (should only have one row)
            metric_row = metrics.iloc[0].to_dict()

            # Rename columns to match expected format
            renamed_metrics = {}
            for key, value in metric_row.items():
                # Map column names from evaluate_forecasts format to expected format
                if key == 'MSE[mean]':
                    renamed_metrics['MSE_mean'] = value
                elif key == 'MSE[0.5]':
                    renamed_metrics['MSE_median'] = value
                elif key == 'MAE[0.5]':
                    renamed_metrics['MAE_median'] = value
                elif key == 'MASE[0.5]':
                    renamed_metrics['MASE_median'] = value
                elif key == 'MAPE[0.5]':
                    renamed_metrics['MAPE_median'] = value
                elif key == 'sMAPE[0.5]':
                    renamed_metrics['sMAPE_median'] = value
                elif key == 'RMSE[mean]':
                    renamed_metrics['RMSE_mean'] = value
                elif key == 'NRMSE[mean]':
                    renamed_metrics['NRMSE_mean'] = value
                elif key == 'ND[0.5]':
                    renamed_metrics['ND_median'] = value
                else:
                    # Keep other columns as-is
                    renamed_metrics[key] = value

            dataset_metrics[canonical_name] = renamed_metrics
            print(f"✓ Loaded metrics for {canonical_name} ({dataset_name_raw})")
    except Exception as e:
        print(f"✗ Error reading {metrics_file}: {e}")

# Update the summary DataFrame
for idx, dataset in enumerate(datasets):
    if dataset in dataset_metrics:
        metrics = dataset_metrics[dataset]
        summary_df.loc[idx, 'status'] = 'success'
        for key, value in metrics.items():
            if key in summary_df.columns:
                summary_df.loc[idx, key] = value
        print(f"Updated {dataset}")
    else:
        print(f"No metrics found for {dataset}")

# Save the updated summary
summary_df.to_csv(output_csv, index=False)
print(f"\n✓ Summary CSV updated: {output_csv}")
print(f"\nSuccessful datasets: {(summary_df['status'] == 'success').sum()}/{len(summary_df)}")
