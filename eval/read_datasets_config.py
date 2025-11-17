#!/usr/bin/env python3
"""
Read dataset configuration from forecast_datasets.xlsx and output as JSON.

This script is used by SLURM evaluation scripts to load dataset names,
prediction lengths, and frequencies in the correct order.

Output format:
[
    {
        "display_name": "M1_Monthly",
        "dataset_name": "m1_monthly",
        "prediction_length": 18,
        "frequency": "M"
    },
    ...
]
"""

import pandas as pd
import json
import sys
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    excel_path = script_dir / "forecast_datasets.xlsx"

    if not excel_path.exists():
        print(f"ERROR: Excel file not found at {excel_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)

        # Map actual Excel columns to expected names
        # Actual columns: 'Dataset', 'Domain', 'Frequency', 'Number of Series', 'Prediction Length'
        column_mapping = {
            'Dataset': 'display_name',
            'Prediction Length': 'prediction_length',
            'Frequency': 'frequency'
        }

        # Check if required columns exist
        if 'Dataset' not in df.columns or 'Prediction Length' not in df.columns:
            print(f"ERROR: Missing required columns in Excel file", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

        # Convert to list of dictionaries
        datasets = []
        for _, row in df.iterrows():
            # Skip rows with missing essential data
            if pd.isna(row['Dataset']) or pd.isna(row['Prediction Length']):
                continue

            # Create dataset_name from display_name (lowercase with underscores)
            display_name = str(row['Dataset']).strip()

            # Special mapping for datasets with different cached names
            # This maps display names to actual cached dataset directory names
            dataset_name_mapping = {
                'Aus. Elec. Demand': 'australian_electricity_demand',
                'Australia Weather': 'weather',
                'Bitcoin': 'bitcoin_with_missing',
                'Carparts': 'car_parts_with_missing',
                'CIF 2016': 'cif_2016_12',  # Use 12-month version
                'KDD Cup 2018': 'kdd_cup_2018_with_missing',
                'M3 Monthly': 'monash_m3_monthly',
                'M3 Other': 'monash_m3_other',
                'NN5 Daily': 'nn5_daily_with_missing',
                'NN5 Weekly': 'nn5_weekly',
                'Rideshare': 'rideshare_with_missing',
                'Saugeen River Flow': 'saugeenday',
                'Sunspot': 'sunspot_with_missing',
                'Temperature Rain': 'temperature_rain_with_missing',
                'Vehicle Trips': 'vehicle_trips_with_missing',
            }

            # Use mapping if available, otherwise generate from display name
            if display_name in dataset_name_mapping:
                dataset_name = dataset_name_mapping[display_name]
            else:
                dataset_name = display_name.lower().replace(' ', '_').replace('.', '').replace('-', '_')

            dataset_config = {
                'display_name': display_name,
                'dataset_name': dataset_name,
                'prediction_length': int(row['Prediction Length'])
            }

            # Add frequency if available
            if 'Frequency' in df.columns and pd.notna(row['Frequency']):
                dataset_config['frequency'] = str(row['Frequency']).strip()

            datasets.append(dataset_config)

        # Output as JSON to stdout
        print(json.dumps(datasets))

    except Exception as e:
        print(f"ERROR: Failed to read Excel file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
