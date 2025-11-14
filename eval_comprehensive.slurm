#!/bin/bash
#SBATCH --job-name=moirai_comprehensive_eval
#SBATCH --output=logs/eval_comprehensive_%j.out
#SBATCH --error=logs/eval_comprehensive_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --mail-type=END
#SBATCH --mail-user=jh1161@princeton.edu

# Comprehensive evaluation script for Moirai models
#
# Dataset configuration is loaded from eval_confs/forecast_datasets.xlsx
# This file specifies dataset names, prediction lengths, and evaluation order
#
# Features:
# - Automatically loads datasets and prediction lengths from Excel file
# - Maintains dataset order as specified in the Excel file
# - Handles partial failures: extracts MAE metric even if other metrics fail
# - Saves results to CSV with comprehensive metrics
#
# Usage:
#   Official model: sbatch eval_comprehensive.slurm
#   Custom model:   sbatch --export=MODEL_PATH=/path/to/checkpoint eval_comprehensive.slurm

# Print job information
echo "=========================================="
echo "Comprehensive Moirai Evaluation"
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

# Model configuration
if [ -n "$MODEL_PATH" ]; then
    # Custom model path provided
    echo "=========================================="
    echo "Using CUSTOM model checkpoint"
    echo "=========================================="
    echo "Model Path: $MODEL_PATH"
    echo "Patch Size: $PATCH_SIZE"
    echo "Context Length: $CONTEXT_LENGTH"
    echo "Batch Size: $BATCH_SIZE"
    echo "=========================================="
    MODEL_TYPE="custom"
    MODEL_NAME=$(basename $MODEL_PATH)
    # For custom models, you'll need to create a config or modify the eval command
    # This will be handled in the Python script section
else
    # Official model from HuggingFace
    MODEL_VERSION=${MODEL_VERSION:-1.1}  # 1.0 or 1.1

    if [ "$MODEL_VERSION" == "1.1" ]; then
        MODEL_CONFIG="moirai_1.1_R_small"
        MODEL_NAME="moirai-1.1-R-small"
    elif [ "$MODEL_VERSION" == "1.0" ]; then
        MODEL_CONFIG="moirai_1.0_R_small"
        MODEL_NAME="moirai-1.0-R-small"
    else
        echo "ERROR: Invalid MODEL_VERSION. Use 1.0 or 1.1"
        exit 1
    fi

    echo "=========================================="
    echo "Using OFFICIAL model from HuggingFace"
    echo "=========================================="
    echo "Model: Salesforce/${MODEL_NAME}"
    echo "Patch Size: $PATCH_SIZE"
    echo "Context Length: $CONTEXT_LENGTH"
    echo "Batch Size: $BATCH_SIZE"
    echo "=========================================="
    MODEL_TYPE="official"
fi

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
RESULTS_DIR="eval_results_${MODEL_TYPE}_${MODEL_NAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Initialize CSV file
CSV_FILE="$RESULTS_DIR/evaluation_metrics.csv"
echo "dataset,MSE_mean,MSE_median,MAE_median,MASE_median,MAPE_median,sMAPE_median,MSIS,RMSE_mean,NRMSE_mean,ND_median,mean_weighted_sum_quantile_loss,status" > "$CSV_FILE"

# Track success/failure
declare -a SUCCESSFUL_DATASETS
declare -a FAILED_DATASETS

# Function to extract metrics from output and append to CSV
# Handles partial failures - if MAE is present, extract it even if other metrics fail
extract_metrics() {
    local dataset_name=$1
    local output_file=$2

    # Look for the metrics line (starts with "None" and contains numbers)
    local metrics_line=$(grep -E "^None\s+[0-9]" "$output_file" | tail -1)

    if [ -n "$metrics_line" ]; then
        # Parse the metrics
        read -r none mse_mean mse_median mae_median mase_median mape_median smape_median msis rmse_mean nrmse_mean nd_median mean_wql <<< "$metrics_line"
        echo "$dataset_name,$mse_mean,$mse_median,$mae_median,$mase_median,$mape_median,$smape_median,$msis,$rmse_mean,$nrmse_mean,$nd_median,$mean_wql,success" >> "$CSV_FILE"
        return 0
    else
        # Check if we can at least extract MAE from the output
        # Look for lines that might contain MAE even if full metrics table failed
        local mae_value=$(grep -i "mae" "$output_file" | grep -oE "[0-9]+\.[0-9]+" | head -1)

        if [ -n "$mae_value" ]; then
            # Found MAE, save it with other metrics as empty
            echo "$dataset_name,,,$mae_value,,,,,,,partial_success" >> "$CSV_FILE"
            echo "  Note: Only MAE metric extracted for $dataset_name"
            return 0
        else
            echo "$dataset_name,,,,,,,,,,,failed" >> "$CSV_FILE"
            return 1
        fi
    fi
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

    if [ "$MODEL_TYPE" == "official" ]; then
        # Official model evaluation with prediction length
        python -m cli.eval \
          run_name=eval_${MODEL_TYPE}_${MODEL_NAME}_${dataset_name} \
          model=$MODEL_CONFIG \
          model.patch_size=$PATCH_SIZE \
          model.context_length=$CONTEXT_LENGTH \
          batch_size=$BATCH_SIZE \
          data=monash_cached \
          data.dataset_name=$dataset_name \
          data.prediction_length=$prediction_length > "$TEMP_OUTPUT" 2>&1
    else
        # Custom model evaluation with prediction length
        python -m cli.eval \
          run_name=eval_${MODEL_TYPE}_${MODEL_NAME}_${dataset_name} \
          model=moirai_lightning_ckpt \
          model.checkpoint_path=$MODEL_PATH \
          model.patch_size=$PATCH_SIZE \
          model.context_length=$CONTEXT_LENGTH \
          batch_size=$BATCH_SIZE \
          data=monash_cached \
          data.dataset_name=$dataset_name \
          data.prediction_length=$prediction_length > "$TEMP_OUTPUT" 2>&1
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $display_name completed successfully"
        SUCCESSFUL_DATASETS+=("$display_name")
        extract_metrics "$display_name" "$TEMP_OUTPUT"
    else
        echo "✗ $display_name failed (exit code: $EXIT_CODE)"
        FAILED_DATASETS+=("$display_name")
        # Try to extract MAE even on failure
        extract_metrics "$display_name" "$TEMP_OUTPUT"
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

exit 0
