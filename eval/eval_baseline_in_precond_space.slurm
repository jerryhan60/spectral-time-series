#!/bin/bash
#SBATCH --job-name=baseline_precond_space
#SBATCH --output=logs/eval_baseline_precond_space_%j.out
#SBATCH --error=logs/eval_baseline_precond_space_%j.err
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

# Comprehensive evaluation script to compare BASELINE vs PRECONDITIONED models
#
# This script evaluates a BASELINE model by:
# 1. Generating predictions from baseline model (in original space)
# 2. Applying preconditioning to both predictions AND ground truth
# 3. Computing metrics in the preconditioned/transformed space
#
# This enables fair comparison with preconditioned models that directly
# output predictions in the transformed space.
#
# Usage:
#   sbatch --export=MODEL_PATH=/path/to/baseline.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 eval_baseline_in_precond_space.slurm
#
# Parameters:
#   MODEL_PATH: Path to baseline model checkpoint (REQUIRED)
#   PRECOND_TYPE: chebyshev or legendre (default: chebyshev)
#   PRECOND_DEGREE: Polynomial degree (default: 5)
#   PATCH_SIZE: Override default patch size (default: frequency-based)
#   CONTEXT_LENGTH: Context window (default: 1000)
#   BATCH_SIZE: Batch size (default: 32)

# Print job information
echo "=========================================="
echo "Baseline Model in Preconditioned Space Evaluation"
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

# Force HuggingFace to use offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HuggingFace offline mode enabled"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Configuration
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1000}
BATCH_SIZE=${BATCH_SIZE:-32}
PRECOND_TYPE=${PRECOND_TYPE:-chebyshev}
PRECOND_DEGREE=${PRECOND_DEGREE:-5}

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH environment variable is required"
    echo "Usage: sbatch --export=MODEL_PATH=/path/to/baseline.ckpt eval_baseline_in_precond_space.slurm"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found at: $MODEL_PATH"
    exit 1
fi

MODEL_NAME=$(basename $MODEL_PATH .ckpt)

echo "=========================================="
echo "Configuration:"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "Context Length: $CONTEXT_LENGTH"
echo "Batch Size: $BATCH_SIZE"
echo "Preconditioning Type: $PRECOND_TYPE"
echo "Preconditioning Degree: $PRECOND_DEGREE"
echo "Evaluation Mode: BASELINE predictions -> PRECONDITIONED space"
echo "=========================================="
echo ""

# Load dataset configuration from Excel file
echo "Loading dataset configuration from forecast_datasets.xlsx..."
DATASETS_JSON=$(/scratch/gpfs/EHAZAN/jh1161/eval/read_datasets_config.py)

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to read dataset configuration from Excel file"
    exit 1
fi

# Parse JSON into arrays
DATASET_DISPLAY_NAMES=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([d['display_name'].replace(' ', '_') for d in data]))"))
DATASET_NAMES=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([d['dataset_name'] for d in data]))"))
PREDICTION_LENGTHS=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([str(d['prediction_length']) for d in data]))"))
FREQUENCIES=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([d.get('frequency', 'UNKNOWN') for d in data]))"))

echo "Loaded ${#DATASET_NAMES[@]} datasets from configuration file"

# Output directory for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/scratch/gpfs/EHAZAN/jh1161/eval/outputs/eval_baseline_precond_space_${MODEL_NAME}_${PRECOND_TYPE}_d${PRECOND_DEGREE}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Initialize CSV file with standard format
CSV_FILE="$RESULTS_DIR/evaluation_metrics_baseline_in_precond_space.csv"
echo "dataset,MSE[mean],MSE[0.5],MAE[0.5],MASE[0.5],MAPE[0.5],sMAPE[0.5],MSIS,RMSE[mean],NRMSE[mean],ND[0.5],mean_weighted_sum_quantile_loss,status" > "$CSV_FILE"

# Track success/failure
declare -a SUCCESSFUL_DATASETS
declare -a FAILED_DATASETS

# Function to extract metrics from output and append to CSV
extract_metrics() {
    local dataset_name=$1
    local output_file=$2

    # Look for metrics in the new standard format
    local mse_mean=$(grep "MSE\[mean\]:" "$output_file" | grep -oE "[0-9]+\.?[0-9]*" | head -1)
    local mse_median=$(grep "MSE\[0\.5\]:" "$output_file" | grep -oE "[0-9]+\.?[0-9]*" | head -1)
    local mae_median=$(grep "MAE\[0\.5\]:" "$output_file" | grep -oE "[0-9]+\.?[0-9]*" | head -1)
    local rmse_mean=$(grep "RMSE\[mean\]:" "$output_file" | grep -oE "[0-9]+\.?[0-9]*" | head -1)

    if [ -n "$mae_median" ]; then
        # At least MAE found - write with standard format (other metrics are NaN for baseline)
        echo "$dataset_name,$mse_mean,$mse_median,$mae_median,,,,,,$rmse_mean,,,,success" >> "$CSV_FILE"
        return 0
    else
        echo "$dataset_name,,,,,,,,,,,,,failed" >> "$CSV_FILE"
        return 1
    fi
}

# Run evaluation on all datasets
echo "=========================================="
echo "Starting Evaluation on All Datasets"
echo "=========================================="
TOTAL_DATASETS=${#DATASET_NAMES[@]}
CURRENT=0

# Iterate through datasets
for i in "${!DATASET_NAMES[@]}"; do
    dataset_name="${DATASET_NAMES[$i]}"
    display_name="${DATASET_DISPLAY_NAMES[$i]}"
    prediction_length="${PREDICTION_LENGTHS[$i]}"
    frequency="${FREQUENCIES[$i]}"
    CURRENT=$((CURRENT + 1))

    # Determine patch size based on frequency (per Moirai paper Appendix B.1)
    case "$frequency" in
        Y|Q)
            DATASET_PATCH_SIZE=8
            ;;
        M|W|D|H)
            DATASET_PATCH_SIZE=32
            ;;
        *T|*S)
            DATASET_PATCH_SIZE=64
            ;;
        *)
            DATASET_PATCH_SIZE=${PATCH_SIZE:-32}
            ;;
    esac

    echo ""
    echo "[$CURRENT/$TOTAL_DATASETS] Evaluating: $display_name"
    echo "  Dataset: $dataset_name"
    echo "  Frequency: $frequency"
    echo "  Prediction Length: $prediction_length"
    echo "  Patch Size: $DATASET_PATCH_SIZE"
    echo "  Preconditioning: $PRECOND_TYPE degree $PRECOND_DEGREE"
    echo "  $(date)"

    # Create output file for this dataset
    TEMP_OUTPUT="$RESULTS_DIR/${display_name}_output.txt"

    # Run evaluation
    # NOTE: precond_type and precond_degree are top-level config params (not in base struct)
    # They need the + prefix to be added to the config
    python -m cli.eval_baseline_in_precond_space \
      run_name=eval_baseline_precond_${dataset_name} \
      model=moirai_lightning_ckpt \
      model.checkpoint_path=$MODEL_PATH \
      model.patch_size=$DATASET_PATCH_SIZE \
      model.context_length=$CONTEXT_LENGTH \
      +precond_type=$PRECOND_TYPE \
      +precond_degree=$PRECOND_DEGREE \
      batch_size=$BATCH_SIZE \
      data=monash_cached \
      data.dataset_name=$dataset_name \
      data.prediction_length=$prediction_length > "$TEMP_OUTPUT" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ $display_name completed successfully"
        SUCCESSFUL_DATASETS+=("$display_name")
        extract_metrics "$display_name" "$TEMP_OUTPUT"
    else
        echo "  ✗ $display_name failed (exit code: $EXIT_CODE)"
        FAILED_DATASETS+=("$display_name")
        extract_metrics "$display_name" "$TEMP_OUTPUT"
        echo "  Error output saved to: $TEMP_OUTPUT"
    fi

    echo "  Progress: $CURRENT/$TOTAL_DATASETS datasets completed"
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
echo "NOTE: These metrics compare BASELINE model predictions vs ground truth,"
echo "both transformed into the preconditioned space for fair comparison with"
echo "preconditioned models."
echo ""

exit 0
