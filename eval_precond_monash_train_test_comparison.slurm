#!/bin/bash
#SBATCH --job-name=eval_train_test
#SBATCH --output=logs/eval_train_test_%j.out
#SBATCH --error=logs/eval_train_test_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --mail-type=END
#SBATCH --mail-user=jh1161@princeton.edu

# Evaluate preconditioned model on BOTH train and test splits of monthly datasets
# This provides a sanity check to compare train vs test performance

# Print job information
echo "=========================================="
echo "Train vs Test Evaluation Comparison"
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

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Checkpoint path
CHECKPOINT_PATH="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/checkpoints/last.ckpt"

# Configuration
PATCH_SIZE=${PATCH_SIZE:-32}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1000}
BATCH_SIZE=${BATCH_SIZE:-32}

# Preconditioning parameters (MUST match training config)
PRECOND_TYPE=${PRECOND_TYPE:-chebyshev}
PRECOND_DEGREE=${PRECOND_DEGREE:-5}

echo "=========================================="
echo "Configuration:"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Patch Size: $PATCH_SIZE"
echo "Context Length: $CONTEXT_LENGTH"
echo "Batch Size: $BATCH_SIZE"
echo ""
echo "Preconditioning Config:"
echo "  Type: $PRECOND_TYPE"
echo "  Degree: $PRECOND_DEGREE"
echo "  Reverse during eval: YES (automatic)"
echo "=========================================="
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    exit 1
fi

# Extract checkpoint name for run naming
CKPT_NAME="precond_$(basename $(dirname $(dirname $CHECKPOINT_PATH)))_$(basename $CHECKPOINT_PATH .ckpt)"

# Monthly datasets (these are out-of-sample during pretraining)
MONTHLY_DATASETS=("tourism_monthly" "m1_monthly" "monash_m3_monthly" "m4_monthly")

# Function to run evaluation
run_eval() {
    local ds=$1
    local split=$2
    local data_config=$3

    echo ""
    echo ">>> Evaluating on dataset: $ds (${split} split)"
    echo "$(date)"

    python -m cli.eval \
      run_name=eval_${CKPT_NAME}_${ds}_${split} \
      model=moirai_precond_ckpt \
      model.checkpoint_path=$CHECKPOINT_PATH \
      model.patch_size=$PATCH_SIZE \
      model.context_length=$CONTEXT_LENGTH \
      model.enable_preconditioning=true \
      model.precondition_type=$PRECOND_TYPE \
      model.precondition_degree=$PRECOND_DEGREE \
      batch_size=$BATCH_SIZE \
      data=$data_config \
      data.dataset_name=$ds

    if [ $? -eq 0 ]; then
        echo "✓ $ds ($split) completed successfully"
        return 0
    else
        echo "✗ $ds ($split) failed"
        return 1
    fi
}

# Run evaluation on all monthly datasets - BOTH train and test splits
echo ""
echo "=========================================="
echo "Evaluating on MONTHLY datasets"
echo "Comparing TRAIN vs TEST performance"
echo "=========================================="

for ds in "${MONTHLY_DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Dataset: $ds"
    echo "========================================"

    # Evaluate on TRAIN split
    run_eval "$ds" "train" "monash_cached_train"
    TRAIN_STATUS=$?

    # Evaluate on TEST split
    run_eval "$ds" "test" "monash_cached"
    TEST_STATUS=$?

    echo ""
    if [ $TRAIN_STATUS -eq 0 ] && [ $TEST_STATUS -eq 0 ]; then
        echo "✓✓ $ds: Both train and test completed successfully"
    elif [ $TRAIN_STATUS -eq 0 ]; then
        echo "✓✗ $ds: Train succeeded, test failed"
    elif [ $TEST_STATUS -eq 0 ]; then
        echo "✗✓ $ds: Train failed, test succeeded"
    else
        echo "✗✗ $ds: Both train and test failed"
    fi
done

echo ""
echo "=========================================="
echo "All evaluations completed"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Monthly datasets evaluated: ${#MONTHLY_DATASETS[@]}"
echo "  - Splits per dataset: 2 (train + test)"
echo "  - Total evaluations: $((${#MONTHLY_DATASETS[@]} * 2))"
echo ""
echo "Results saved in:"
echo "  - Train: uni2ts/outputs/eval/monash_cached_train/*/"
echo "  - Test:  uni2ts/outputs/eval/monash_cached/*/"
echo ""
echo "=========================================="
echo "INTERPRETATION GUIDE:"
echo "=========================================="
echo ""
echo "These monthly datasets were in TRAIN_TEST_GROUP during pretraining:"
echo "  - Model saw ONLY the train split during pretraining"
echo "  - Test split is completely held-out (out-of-sample)"
echo ""
echo "Expected behavior:"
echo "  - Train error should be LOWER (model saw this data)"
echo "  - Test error should be HIGHER (model never saw this data)"
echo "  - Large gap suggests potential overfitting"
echo "  - Similar errors suggest good generalization"
echo ""
echo "Compare metrics (MSE, MAE, etc.) between train and test:"
echo "  python -m cli.compare_results \\"
echo "    --train-path outputs/eval/monash_cached_train/ \\"
echo "    --test-path outputs/eval/monash_cached/"
echo "=========================================="

exit 0
