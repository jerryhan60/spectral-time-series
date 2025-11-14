#!/scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/python
"""
Helper script to read forecast_datasets.xlsx and output dataset configuration
for the eval_comprehensive.slurm script.
"""
import pandas as pd
import sys
import json

def main():
    xlsx_path = "/scratch/gpfs/EHAZAN/jh1161/eval_confs/forecast_datasets.xlsx"

    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}", file=sys.stderr)
        sys.exit(1)

    # Mapping from display names in Excel to dataset names in lotsa_v1
    # Based on the existing DATASETS mapping in the slurm script
    dataset_mapping = {
        "M1 Monthly": "m1_monthly",
        "M3 Monthly": "monash_m3_monthly",
        "M3 Other": "monash_m3_other",
        "M4 Monthly": "m4_monthly",
        "M4 Weekly": "m4_weekly",
        "M4 Daily": "m4_daily",
        "M4 Hourly": "m4_hourly",
        "Tourism Quarterly": "tourism_quarterly",
        "Tourism Monthly": "tourism_monthly",
        "CIF 2016": "cif_2016_12",
        "Aus. Elec. Demand": "australian_electricity_demand",
        "Bitcoin": "bitcoin_with_missing",
        "Pedestrian Counts": "pedestrian_counts",
        "Vehicle Trips": "vehicle_trips_with_missing",
        "KDD Cup 2018": "kdd_cup_2018_with_missing",
        "Australia Weather": "weather",
        "NN5 Daily": "nn5_daily_with_missing",
        "NN5 Weekly": "nn5_weekly",
        "Carparts": "car_parts_with_missing",
        "FRED-MD": "fred_md",
        "Traffic Hourly": "traffic_hourly",
        "Traffic Weekly": "traffic_weekly",
        "Rideshare": "rideshare_with_missing",
        "Hospital": "hospital",
        "COVID Deaths": "covid_deaths",
        "Temperature Rain": "temperature_rain_with_missing",
        "Sunspot": "sunspot_with_missing",
        "Saugeen River Flow": "saugeenday",
        "US Births": "us_births",
    }

    # Create output list maintaining the order from the Excel file
    datasets_config = []

    for _, row in df.iterrows():
        display_name = row['Dataset']
        pred_length = int(row['Prediction Length'])

        if display_name in dataset_mapping:
            dataset_name = dataset_mapping[display_name]
            datasets_config.append({
                'display_name': display_name,
                'dataset_name': dataset_name,
                'prediction_length': pred_length
            })
        else:
            print(f"Warning: No mapping found for dataset '{display_name}'", file=sys.stderr)

    # Output as JSON for easy parsing in bash
    print(json.dumps(datasets_config))

if __name__ == "__main__":
    main()
