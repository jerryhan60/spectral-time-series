#!/bin/bash
#SBATCH --job-name=eval_precond
#SBATCH --output=logs/eval_precond_%j.out
#SBATCH --error=logs/eval_precond_%j.err
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

# Evaluate preconditioned model on Monash datasets
# IMPORTANT: This model was trained WITH preconditioning
# Evaluation will automatically reverse preconditioning for proper comparison

# Print job information
echo "=========================================="
echo "Preconditioned Model Evaluation"
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

# Datasets to evaluate
# Yearly datasets
YEARLY_DATASETS=("tourism_yearly" "m1_yearly" "monash_m3_yearly" "m4_yearly")

# Quarterly datasets
QUARTERLY_DATASETS=("tourism_quarterly" "m1_quarterly" "monash_m3_quarterly" "m4_quarterly")

# Monthly datasets
MONTHLY_DATASETS=("tourism_monthly" "m1_monthly" "monash_m3_monthly" "m4_monthly")

# Run evaluation on all yearly datasets
echo ""
echo "=========================================="
echo "Evaluating on YEARLY datasets"
echo "=========================================="
for ds in "${YEARLY_DATASETS[@]}"; do
    echo ""
    echo ">>> Evaluating on dataset: $ds"
    echo "$(date)"

    python -m cli.eval \
      run_name=eval_${CKPT_NAME}_${ds} \
      model=moirai_precond_ckpt \
      model.checkpoint_path=$CHECKPOINT_PATH \
      model.patch_size=$PATCH_SIZE \
      model.context_length=$CONTEXT_LENGTH \
      model.enable_preconditioning=true \
      model.precondition_type=$PRECOND_TYPE \
      model.precondition_degree=$PRECOND_DEGREE \
      batch_size=$BATCH_SIZE \
      data=monash_cached \
      data.dataset_name=$ds

    if [ $? -eq 0 ]; then
        echo "✓ $ds completed successfully"
    else
        echo "✗ $ds failed"
    fi
done

# Run evaluation on all quarterly datasets
echo ""
echo "=========================================="
echo "Evaluating on QUARTERLY datasets"
echo "=========================================="
for ds in "${QUARTERLY_DATASETS[@]}"; do
    echo ""
    echo ">>> Evaluating on dataset: $ds"
    echo "$(date)"

    python -m cli.eval \
      run_name=eval_${CKPT_NAME}_${ds} \
      model=moirai_precond_ckpt \
      model.checkpoint_path=$CHECKPOINT_PATH \
      model.patch_size=$PATCH_SIZE \
      model.context_length=$CONTEXT_LENGTH \
      model.enable_preconditioning=true \
      model.precondition_type=$PRECOND_TYPE \
      model.precondition_degree=$PRECOND_DEGREE \
      batch_size=$BATCH_SIZE \
      data=monash_cached \
      data.dataset_name=$ds

    if [ $? -eq 0 ]; then
        echo "✓ $ds completed successfully"
    else
        echo "✗ $ds failed"
    fi
done

# Run evaluation on all monthly datasets
echo ""
echo "=========================================="
echo "Evaluating on MONTHLY datasets"
echo "=========================================="
for ds in "${MONTHLY_DATASETS[@]}"; do
    echo ""
    echo ">>> Evaluating on dataset: $ds"
    echo "$(date)"

    python -m cli.eval \
      run_name=eval_${CKPT_NAME}_${ds} \
      model=moirai_precond_ckpt \
      model.checkpoint_path=$CHECKPOINT_PATH \
      model.patch_size=$PATCH_SIZE \
      model.context_length=$CONTEXT_LENGTH \
      model.enable_preconditioning=true \
      model.precondition_type=$PRECOND_TYPE \
      model.precondition_degree=$PRECOND_DEGREE \
      batch_size=$BATCH_SIZE \
      data=monash_cached \
      data.dataset_name=$ds

    if [ $? -eq 0 ]; then
        echo "✓ $ds completed successfully"
    else
        echo "✗ $ds failed"
    fi
done

echo ""
echo "=========================================="
echo "All evaluations completed"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Yearly datasets: ${#YEARLY_DATASETS[@]}"
echo "  - Quarterly datasets: ${#QUARTERLY_DATASETS[@]}"
echo "  - Monthly datasets: ${#MONTHLY_DATASETS[@]}"
echo "  - Total: $((${#YEARLY_DATASETS[@]} + ${#QUARTERLY_DATASETS[@]} + ${#MONTHLY_DATASETS[@]}))"
echo ""
echo "Results saved in: uni2ts/outputs/eval/monash_cached/*/"
echo ""
echo "=========================================="
echo "IMPORTANT NOTE:"
echo "=========================================="
echo "This model was trained WITH preconditioning, and the evaluation"
echo "code NOW PROPERLY reverses preconditioning automatically!"
echo ""
echo "The evaluation pipeline:"
echo "  1. Applies preconditioning to test data (same as training)"
echo "  2. Runs model inference in preconditioned space"
echo "  3. Reverses preconditioning on predictions (back to original scale)"
echo "  4. Computes metrics in original scale for fair comparison"
echo ""
echo "Metrics are now directly comparable to non-preconditioned models."
echo "=========================================="

exit 0
