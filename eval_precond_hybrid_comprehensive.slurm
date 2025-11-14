#!/bin/bash
#SBATCH --job-name=moirai_hybrid_eval
#SBATCH --output=logs/eval_hybrid_comprehensive_%j.out
#SBATCH --error=logs/eval_hybrid_comprehensive_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=hazan_intern
#SBATCH --mail-type=END
#SBATCH --mail-user=jh1161@princeton.edu

# Comprehensive hybrid evaluation script
# Evaluates HYBRID model combining base + preconditioned predictions
#
# Dataset configuration is loaded from eval_confs/forecast_datasets.xlsx
#
# Usage:
#   sbatch --export=BASE_MODEL_PATH=/path/to/base.ckpt,PRECOND_MODEL_PATH=/path/to/precond.ckpt eval_precond_hybrid_comprehensive.slurm
#
# Optional parameters:
#   BASE_MODEL_TYPE=official (default) or custom
#   BASE_MODEL_VERSION=1.1 (default) or 1.0 (only if BASE_MODEL_TYPE=official)
#   PRECOND_TYPE=chebyshev (default) or legendre
#   PRECOND_DEGREE=5 (default)
#   PATCH_SIZE=32 (default)
#   CONTEXT_LENGTH=1000 (default)
#   BATCH_SIZE=32 (default)

# Print job information
echo "=========================================="
echo "Comprehensive Hybrid Model Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Change to project directory
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Load environment
echo "Activating virtual environment..."
source venv/bin/activate

# Force HuggingFace to use offline mode (skip online checks)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HuggingFace offline mode enabled"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Configuration
PATCH_SIZE=${PATCH_SIZE:-32}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1000}
BATCH_SIZE=${BATCH_SIZE:-32}
PRECOND_TYPE=${PRECOND_TYPE:-chebyshev}
PRECOND_DEGREE=${PRECOND_DEGREE:-5}
BASE_MODEL_TYPE=${BASE_MODEL_TYPE:-official}

# Check if preconditioned model path is provided
if [ -z "$PRECOND_MODEL_PATH" ]; then
    echo "ERROR: PRECOND_MODEL_PATH environment variable is required"
    echo "Usage: sbatch --export=PRECOND_MODEL_PATH=/path/to/precond.ckpt eval_precond_hybrid_comprehensive.slurm"
    exit 1
fi

if [ ! -f "$PRECOND_MODEL_PATH" ]; then
    echo "ERROR: Preconditioned model checkpoint not found at: $PRECOND_MODEL_PATH"
    exit 1
fi

PRECOND_MODEL_NAME=$(basename $PRECOND_MODEL_PATH .ckpt)

# Base model configuration
if [ "$BASE_MODEL_TYPE" == "custom" ]; then
    if [ -z "$BASE_MODEL_PATH" ]; then
        echo "ERROR: BASE_MODEL_PATH environment variable is required when BASE_MODEL_TYPE=custom"
        exit 1
    fi
    if [ ! -f "$BASE_MODEL_PATH" ]; then
        echo "ERROR: Base model checkpoint not found at: $BASE_MODEL_PATH"
        exit 1
    fi
    BASE_MODEL_NAME=$(basename $BASE_MODEL_PATH .ckpt)
    BASE_MODEL_CONFIG="moirai_lightning_ckpt"
else
    # Official model from HuggingFace
    BASE_MODEL_VERSION=${BASE_MODEL_VERSION:-1.1}
    if [ "$BASE_MODEL_VERSION" == "1.1" ]; then
        BASE_MODEL_CONFIG="moirai_1.1_R_small"
        BASE_MODEL_NAME="moirai-1.1-R-small"
    elif [ "$BASE_MODEL_VERSION" == "1.0" ]; then
        BASE_MODEL_CONFIG="moirai_1.0_R_small"
        BASE_MODEL_NAME="moirai-1.0-R-small"
    else
        echo "ERROR: Invalid BASE_MODEL_VERSION. Use 1.0 or 1.1"
        exit 1
    fi
fi

echo "=========================================="
echo "Configuration:"
echo "=========================================="
echo "Base Model:"
echo "  Type: $BASE_MODEL_TYPE"
echo "  Name: $BASE_MODEL_NAME"
if [ "$BASE_MODEL_TYPE" == "custom" ]; then
    echo "  Path: $BASE_MODEL_PATH"
fi
echo ""
echo "Preconditioned Model:"
echo "  Name: $PRECOND_MODEL_NAME"
echo "  Path: $PRECOND_MODEL_PATH"
echo "  Type: $PRECOND_TYPE"
echo "  Degree: $PRECOND_DEGREE"
echo ""
echo "General:"
echo "  Patch Size: $PATCH_SIZE"
echo "  Context Length: $CONTEXT_LENGTH"
echo "  Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Load dataset configuration from Excel file
echo "Loading dataset configuration from forecast_datasets.xlsx..."
DATASETS_JSON=$(/scratch/gpfs/EHAZAN/jh1161/read_datasets_config.py)

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to read dataset configuration from Excel file"
    exit 1
fi

# Parse JSON into arrays (maintaining order from Excel file)
DATASET_DISPLAY_NAMES=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([d['display_name'].replace(' ', '_') for d in data]))"))
DATASET_NAMES=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([d['dataset_name'] for d in data]))"))
PREDICTION_LENGTHS=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([str(d['prediction_length']) for d in data]))"))

echo "Loaded ${#DATASET_NAMES[@]} datasets from configuration file"

# Output directory for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="eval_hybrid_results_${BASE_MODEL_NAME}_${PRECOND_MODEL_NAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Initialize CSV file
CSV_FILE="$RESULTS_DIR/evaluation_metrics_hybrid.csv"
echo "dataset,MSE_mean,MSE_median,MAE_median,MASE_median,MAPE_median,sMAPE_median,MSIS,RMSE_mean,NRMSE_mean,ND_median,mean_weighted_sum_quantile_loss,status" > "$CSV_FILE"

# Track success/failure
declare -a SUCCESSFUL_DATASETS
declare -a FAILED_DATASETS

# Function to mark dataset status in CSV (metrics will be aggregated at the end)
mark_dataset_status() {
    local dataset_name=$1
    local status=$2

    # Just mark the status - metrics will be extracted from individual CSV files at the end
    echo "$dataset_name,,,,,,,,,,,${status}" >> "$CSV_FILE"
}

# Run evaluation on all datasets
echo "=========================================="
echo "Starting Evaluation on All Datasets"
echo "=========================================="
TOTAL_DATASETS=${#DATASET_NAMES[@]}
CURRENT=0

# Iterate through datasets in the order from Excel file
for i in "${!DATASET_NAMES[@]}"; do
    dataset_name="${DATASET_NAMES[$i]}"
    display_name="${DATASET_DISPLAY_NAMES[$i]}"
    prediction_length="${PREDICTION_LENGTHS[$i]}"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "[$CURRENT/$TOTAL_DATASETS] Evaluating: $display_name (dataset: $dataset_name, pred_len: $prediction_length)"
    echo "$(date)"

    # Create temporary output file for this dataset
    TEMP_OUTPUT="$RESULTS_DIR/${display_name}_output.txt"

    # Build evaluation command
    if [ "$BASE_MODEL_TYPE" == "custom" ]; then
        # Custom base model
        python -m cli.eval_precond_hybrid \
          run_name=eval_hybrid_${BASE_MODEL_NAME}_${PRECOND_MODEL_NAME}_${dataset_name} \
          model@base_model=moirai_lightning_ckpt \
          base_model.checkpoint_path=$BASE_MODEL_PATH \
          base_model.patch_size=$PATCH_SIZE \
          base_model.context_length=$CONTEXT_LENGTH \
          precond_model.checkpoint_path=$PRECOND_MODEL_PATH \
          precond_model.patch_size=$PATCH_SIZE \
          precond_model.context_length=$CONTEXT_LENGTH \
          precond_model.precondition_type=$PRECOND_TYPE \
          precond_model.precondition_degree=$PRECOND_DEGREE \
          precond_model.reverse_output=false \
          batch_size=$BATCH_SIZE \
          data=monash_cached \
          data.dataset_name=$dataset_name \
          data.prediction_length=$prediction_length > "$TEMP_OUTPUT" 2>&1
    else
        # Official base model
        python -m cli.eval_precond_hybrid \
          run_name=eval_hybrid_${BASE_MODEL_NAME}_${PRECOND_MODEL_NAME}_${dataset_name} \
          model@base_model=$BASE_MODEL_CONFIG \
          base_model.patch_size=$PATCH_SIZE \
          base_model.context_length=$CONTEXT_LENGTH \
          precond_model.checkpoint_path=$PRECOND_MODEL_PATH \
          precond_model.patch_size=$PATCH_SIZE \
          precond_model.context_length=$CONTEXT_LENGTH \
          precond_model.precondition_type=$PRECOND_TYPE \
          precond_model.precondition_degree=$PRECOND_DEGREE \
          precond_model.reverse_output=false \
          batch_size=$BATCH_SIZE \
          data=monash_cached \
          data.dataset_name=$dataset_name \
          data.prediction_length=$prediction_length > "$TEMP_OUTPUT" 2>&1
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $display_name completed successfully"
        SUCCESSFUL_DATASETS+=("$display_name")
        mark_dataset_status "$display_name" "success"
    else
        echo "✗ $display_name failed (exit code: $EXIT_CODE)"
        FAILED_DATASETS+=("$display_name")
        mark_dataset_status "$display_name" "failed"
        # Keep the error output for debugging
        echo "Error output saved to: $TEMP_OUTPUT"
    fi

    # Print progress
    echo "Progress: $CURRENT/$TOTAL_DATASETS datasets completed"
done

echo ""
echo "=========================================="
echo "Evaluation Completed"
echo "=========================================="
echo "End time: $(date)"
echo ""

# Aggregate metrics from individual CSV files
echo "Aggregating metrics from individual CSV files..."
python - <<EOF
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

# Find all individual metrics CSV files
# These are saved by Hydra in timestamped directories like outputs/YYYY-MM-DD/HH-MM-SS/metrics_hybrid.csv
metrics_files = list(Path("outputs").rglob("metrics_hybrid.csv"))
print(f"Found {len(metrics_files)} metrics files")

# Filter to only recent files (from today's run)
from datetime import datetime, timedelta
cutoff_time = datetime.now() - timedelta(hours=24)  # Last 24 hours
recent_files = []
for f in metrics_files:
    mtime = datetime.fromtimestamp(f.stat().st_mtime)
    if mtime > cutoff_time:
        recent_files.append(f)

metrics_files = recent_files
print(f"Found {len(metrics_files)} metrics files from last 24 hours")

# Dataset name mapping (maps internal dataset names to display names)
dataset_name_mapping = {
    'australian_electricity_demand': 'Aus._Elec._Demand',
    'bitcoin_with_missing': 'Bitcoin',
    'car_parts_with_missing': 'Carparts',
    'cif_2016': 'CIF_2016',
    'cif_2016_12': 'CIF_2016',
    'covid_deaths': 'COVID_Deaths',
    'fred_md': 'FRED-MD',
    'hospital': 'Hospital',
    'kdd_cup_2018_with_missing': 'KDD_Cup_2018',
    'm1_monthly': 'M1_Monthly',
    'm3_monthly': 'M3_Monthly',
    'monash_m3_monthly': 'M3_Monthly',
    'm3_other': 'M3_Other',
    'monash_m3_other': 'M3_Other',
    'm4_daily': 'M4_Daily',
    'm4_hourly': 'M4_Hourly',
    'm4_monthly': 'M4_Monthly',
    'm4_weekly': 'M4_Weekly',
    'nn5_daily_with_missing': 'NN5_Daily',
    'nn5_weekly': 'NN5_Weekly',
    'pedestrian_counts': 'Pedestrian_Counts',
    'rideshare_with_missing': 'Rideshare',
    'saugeen_river_flow': 'Saugeen_River_Flow',
    'saugeenday': 'Saugeen_River_Flow',
    'sunspot_with_missing': 'Sunspot',
    'temperature_rain': 'Temperature_Rain',
    'temperature_rain_with_missing': 'Temperature_Rain',
    'tourism_monthly': 'Tourism_Monthly',
    'tourism_quarterly': 'Tourism_Quarterly',
    'traffic_hourly': 'Traffic_Hourly',
    'traffic_weekly': 'Traffic_Weekly',
    'us_births': 'US_Births',
    'vehicle_trips_with_missing': 'Vehicle_Trips',
    'weather': 'Australia_Weather',
}

# Dictionary to store metrics
dataset_metrics = {}

# Process each metrics file
for metrics_file in metrics_files:
    # For hybrid evaluation, Hydra saves to outputs/YYYY-MM-DD/HH-MM-SS/
    # We need to look at the .hydra/config.yaml to get the dataset name
    hydra_config = metrics_file.parent / ".hydra" / "config.yaml"

    if not hydra_config.exists():
        continue

    try:
        # Read the Hydra config to get dataset name
        import yaml
        with open(hydra_config) as f:
            config = yaml.safe_load(f)

        dataset_name_raw = config.get('data', {}).get('dataset_name', '')
        if not dataset_name_raw:
            continue

        canonical_name = dataset_name_mapping.get(dataset_name_raw, dataset_name_raw)

        # Skip if already processed
        if canonical_name in dataset_metrics:
            continue

    except Exception as e:
        print(f"Error reading config for {metrics_file}: {e}")
        continue

    try:
        metrics = pd.read_csv(metrics_file)
        if len(metrics) > 0:
            metric_row = metrics.iloc[0].to_dict()
            renamed_metrics = {}
            has_valid_metric = False
            valid_metric_count = 0

            for key, value in metric_row.items():
                # Check if value is valid (not NaN, not None, not empty string)
                is_valid = pd.notna(value) and value != '' and not (isinstance(value, float) and np.isnan(value))

                if key == 'MSE[mean]':
                    renamed_metrics['MSE_mean'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'MSE[0.5]':
                    renamed_metrics['MSE_median'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'MAE[0.5]':
                    renamed_metrics['MAE_median'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'MASE[0.5]':
                    renamed_metrics['MASE_median'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'MAPE[0.5]':
                    renamed_metrics['MAPE_median'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'sMAPE[0.5]':
                    renamed_metrics['sMAPE_median'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'RMSE[mean]':
                    renamed_metrics['RMSE_mean'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'NRMSE[mean]':
                    renamed_metrics['NRMSE_mean'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'ND[0.5]':
                    renamed_metrics['ND_median'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'MSIS':
                    renamed_metrics['MSIS'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1
                elif key == 'mean_weighted_sum_quantile_loss':
                    renamed_metrics['mean_weighted_sum_quantile_loss'] = value if is_valid else None
                    if is_valid:
                        has_valid_metric = True
                        valid_metric_count += 1

            # Add to dataset_metrics regardless of whether we have valid metrics
            total_expected_metrics = 11
            if valid_metric_count == 0:
                renamed_metrics['status'] = 'all_nan'
                print(f"  {canonical_name}: 0/{total_expected_metrics} valid metrics (all NaN)")
            elif valid_metric_count < total_expected_metrics:
                renamed_metrics['status'] = 'partial_success'
                print(f"  {canonical_name}: {valid_metric_count}/{total_expected_metrics} valid metrics (partial)")
            else:
                renamed_metrics['status'] = 'success'
                print(f"  {canonical_name}: {valid_metric_count}/{total_expected_metrics} valid metrics (success)")

            dataset_metrics[canonical_name] = renamed_metrics
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")

# Define the desired order based on forecast_datasets.xlsx
desired_order = [
    'M1_Monthly',
    'M3_Monthly',
    'M3_Other',
    'M4_Monthly',
    'M4_Weekly',
    'M4_Daily',
    'M4_Hourly',
    'Tourism_Quarterly',
    'Tourism_Monthly',
    'CIF_2016',
    'Aus._Elec._Demand',
    'Bitcoin',
    'Pedestrian_Counts',
    'Vehicle_Trips',
    'KDD_Cup_2018',
    'Australia_Weather',
    'NN5_Daily',
    'NN5_Weekly',
    'Carparts',
    'FRED-MD',
    'Traffic_Hourly',
    'Traffic_Weekly',
    'Rideshare',
    'Hospital',
    'COVID_Deaths',
    'Temperature_Rain',
    'Sunspot',
    'Saugeen_River_Flow',
    'US_Births'
]

# Read and update summary CSV
summary_df = pd.read_csv("$CSV_FILE")
for idx, row in summary_df.iterrows():
    dataset = row['dataset']
    if dataset in dataset_metrics:
        metrics = dataset_metrics[dataset]
        for key, value in metrics.items():
            if key in summary_df.columns:
                # Handle None values by leaving them empty in the CSV
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    summary_df.loc[idx, key] = ''
                else:
                    summary_df.loc[idx, key] = value

# Sort dataframe by desired order
summary_df['sort_order'] = summary_df['dataset'].map({name: i for i, name in enumerate(desired_order)})
summary_df = summary_df.sort_values('sort_order').drop('sort_order', axis=1)

# Save updated CSV
summary_df.to_csv("$CSV_FILE", index=False)
print(f"\nUpdated CSV: $CSV_FILE")
print(f"Successful datasets: {(summary_df['status'] == 'success').sum()}/{len(summary_df)}")
print(f"Partial success datasets: {(summary_df['status'] == 'partial_success').sum()}/{len(summary_df)}")
print(f"All NaN datasets: {(summary_df['status'] == 'all_nan').sum()}/{len(summary_df)}")
EOF

echo ""
echo "Summary:"
echo "  Total datasets: $TOTAL_DATASETS"
echo "  Successful: ${#SUCCESSFUL_DATASETS[@]}"
echo "  Failed: ${#FAILED_DATASETS[@]}"
echo ""

if [ ${#SUCCESSFUL_DATASETS[@]} -gt 0 ]; then
    echo "Successful datasets:"
    for ds in "${SUCCESSFUL_DATASETS[@]}"; do
        echo "  ✓ $ds"
    done
    echo ""
fi

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "Failed datasets:"
    for ds in "${FAILED_DATASETS[@]}"; do
        echo "  ✗ $ds"
    done
    echo ""
fi

echo "Results saved to:"
echo "  CSV metrics: $CSV_FILE"
echo "  Full outputs: $RESULTS_DIR/"
echo ""
echo "NOTE: These are HYBRID metrics combining base + preconditioned model predictions."
echo ""

exit 0
