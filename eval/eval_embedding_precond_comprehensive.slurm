#!/bin/bash
#SBATCH --job-name=eval_embed_precond
#SBATCH --output=logs/eval_embed_precond_%j.out
#SBATCH --error=logs/eval_embed_precond_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup

# ============================================================================
# Comprehensive Evaluation for Embedding-Preconditioned Models
# ============================================================================
#
# This script evaluates models trained with EMBEDDING-LEVEL preconditioning
# on multiple Monash benchmark datasets.
#
# Key difference from data-level preconditioning:
# - Embedding-level: Preconditioning happens INSIDE the model (MoiraiModule.forward)
# - No special reversal needed at evaluation - model outputs are already in original space
#
# Usage:
#   sbatch --export=CHECKPOINT_PATH=/path/to/checkpoint.ckpt eval_embedding_precond_comprehensive.slurm
#
#   # With custom parameters:
#   sbatch --export=CHECKPOINT_PATH=/path/to/ckpt,PATCH_SIZE=32,CONTEXT_LENGTH=1000 eval_embedding_precond_comprehensive.slurm
#
# ============================================================================

echo "=========================================="
echo "Embedding-Preconditioned Model Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Change to project directory
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Load environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Parameters (can be overridden via --export)
CHECKPOINT_PATH=${CHECKPOINT_PATH:-???}
PATCH_SIZE=${PATCH_SIZE:-32}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1000}
NUM_SAMPLES=${NUM_SAMPLES:-100}
BATCH_SIZE=${BATCH_SIZE:-256}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-embed_precond_eval}

# Validate checkpoint
if [ "$CHECKPOINT_PATH" = "???" ]; then
    echo "ERROR: CHECKPOINT_PATH not specified!"
    echo "Usage: sbatch --export=CHECKPOINT_PATH=/path/to/checkpoint.ckpt eval_embedding_precond_comprehensive.slurm"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "=========================================="
echo "Configuration:"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Patch Size: $PATCH_SIZE"
echo "Context Length: $CONTEXT_LENGTH"
echo "Num Samples: $NUM_SAMPLES"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Check embedding preconditioning status in checkpoint
echo "Checking checkpoint for embedding preconditioning..."
python -c "
import torch
ckpt = torch.load('$CHECKPOINT_PATH', map_location='cpu')
hparams = ckpt.get('hyper_parameters', {})
module_kwargs = hparams.get('module_kwargs', {})
print('Module kwargs keys:', list(module_kwargs.keys()))
if 'enable_embedding_preconditioning' in module_kwargs:
    print(f'  enable_embedding_preconditioning: {module_kwargs[\"enable_embedding_preconditioning\"]}')
if 'embedding_precondition_reverse' in module_kwargs:
    print(f'  embedding_precondition_reverse: {module_kwargs[\"embedding_precondition_reverse\"]}')
if 'embedding_precondition_type' in module_kwargs:
    print(f'  embedding_precondition_type: {module_kwargs[\"embedding_precondition_type\"]}')
if 'embedding_precondition_degree' in module_kwargs:
    print(f'  embedding_precondition_degree: {module_kwargs[\"embedding_precondition_degree\"]}')
if 'num_target_variates' in module_kwargs:
    print(f'  num_target_variates: {module_kwargs[\"num_target_variates\"]} (None = all variates)')
"
echo ""

# Define datasets to evaluate (Monash benchmark)
DATASETS=(
    "m1_monthly"
    "m1_quarterly"
    "m1_yearly"
    "m3_monthly"
    "m3_quarterly"
    "m3_yearly"
    "m3_other"
    "m4_hourly"
    "m4_daily"
    "m4_weekly"
    "m4_monthly"
    "m4_quarterly"
    "m4_yearly"
    "tourism_monthly"
    "tourism_quarterly"
    "tourism_yearly"
    "nn5_daily"
    "nn5_weekly"
    "traffic_hourly"
    "traffic_weekly"
    "electricity_hourly"
    "electricity_weekly"
)

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/scratch/gpfs/EHAZAN/jh1161/eval/outputs/${OUTPUT_PREFIX}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Results file
RESULTS_FILE="$OUTPUT_DIR/results.csv"
echo "dataset,mse_mean,mae_0.5,mase,msis,crps,status" > "$RESULTS_FILE"

# Evaluate on each dataset
echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="

TOTAL_DATASETS=${#DATASETS[@]}
CURRENT=0

for DATASET in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL_DATASETS] Evaluating: $DATASET"
    echo "-------------------------------------------"

    # Run evaluation using standard eval.py with checkpoint
    python -m cli.eval \
        model=moirai_embedding_precond_ckpt \
        model.checkpoint_path="$CHECKPOINT_PATH" \
        model.patch_size=$PATCH_SIZE \
        model.context_length=$CONTEXT_LENGTH \
        model.num_samples=$NUM_SAMPLES \
        data=monash_cached \
        data.dataset_name=$DATASET \
        batch_size=$BATCH_SIZE \
        2>&1 | tee "$OUTPUT_DIR/${DATASET}_output.txt"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        # Extract metrics from output
        MSE=$(grep -oP "MSE\[mean\]:\s*\K[\d.]+" "$OUTPUT_DIR/${DATASET}_output.txt" || echo "N/A")
        MAE=$(grep -oP "MAE\[0\.5\]:\s*\K[\d.]+" "$OUTPUT_DIR/${DATASET}_output.txt" || echo "N/A")
        MASE=$(grep -oP "MASE:\s*\K[\d.]+" "$OUTPUT_DIR/${DATASET}_output.txt" || echo "N/A")
        MSIS=$(grep -oP "MSIS:\s*\K[\d.]+" "$OUTPUT_DIR/${DATASET}_output.txt" || echo "N/A")
        CRPS=$(grep -oP "CRPS:\s*\K[\d.]+" "$OUTPUT_DIR/${DATASET}_output.txt" || echo "N/A")

        echo "$DATASET,$MSE,$MAE,$MASE,$MSIS,$CRPS,success" >> "$RESULTS_FILE"
        echo "  Status: SUCCESS"
    else
        echo "$DATASET,N/A,N/A,N/A,N/A,N/A,failed" >> "$RESULTS_FILE"
        echo "  Status: FAILED (exit code: $EXIT_CODE)"
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
cat "$RESULTS_FILE"
echo ""
echo "End time: $(date)"
