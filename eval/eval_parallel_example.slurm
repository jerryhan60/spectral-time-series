#!/bin/bash
#SBATCH --job-name=eval_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # Number of CPU cores for parallel workers
#SBATCH --mem=128G
#SBATCH --time=12:00:00             # Max time for evaluation
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=logs/eval_parallel_%j.out
#SBATCH --error=logs/eval_parallel_%j.err

# ============================================================================
# Parallel Evaluation on GPU Node
# ============================================================================
# This script runs comprehensive evaluation with multiple parallel workers.
# Each worker evaluates a different dataset, speeding up the overall process.
#
# Usage:
#   sbatch eval_parallel_example.slurm
#
# Or with custom settings:
#   sbatch --export=NUM_WORKERS=8,MODE=standard eval_parallel_example.slurm
# ============================================================================

# Load required modules
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate virtual environment
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

# Set environment variables for offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Change to eval directory
cd /scratch/gpfs/EHAZAN/jh1161/eval

# Configuration (can be overridden via --export)
MODE=${MODE:-"standard"}                    # standard, precond, or baseline-precond
NUM_WORKERS=${NUM_WORKERS:-8}               # Number of parallel workers
MODEL_VERSION=${MODEL_VERSION:-"1.1"}       # Model version (1.0 or 1.1)
MODEL_PATH=${MODEL_PATH:-""}                # Path to custom model checkpoint
PRECOND_TYPE=${PRECOND_TYPE:-"chebyshev"}   # Preconditioning type
PRECOND_DEGREE=${PRECOND_DEGREE:-5}         # Preconditioning degree
BATCH_SIZE=${BATCH_SIZE:-32}                # Batch size per evaluation
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1000}      # Context window size

# Print configuration
echo "=========================================="
echo "Parallel Evaluation Configuration"
echo "=========================================="
echo "Mode: $MODE"
echo "Num Workers: $NUM_WORKERS"
echo "Model Version: $MODEL_VERSION"
echo "Model Path: $MODEL_PATH"
echo "Preconditioning Type: $PRECOND_TYPE"
echo "Preconditioning Degree: $PRECOND_DEGREE"
echo "Batch Size: $BATCH_SIZE"
echo "Context Length: $CONTEXT_LENGTH"
echo "=========================================="
echo ""

# Run evaluation based on mode
if [ "$MODE" == "standard" ]; then
    # Standard evaluation
    if [ -z "$MODEL_PATH" ]; then
        # Use official model
        python comprehensive_evaluation_parallel.py \
            --mode standard \
            --model-version $MODEL_VERSION \
            --batch-size $BATCH_SIZE \
            --context-length $CONTEXT_LENGTH \
            --num-workers $NUM_WORKERS
    else
        # Use custom model
        python comprehensive_evaluation_parallel.py \
            --mode standard \
            --model-path "$MODEL_PATH" \
            --batch-size $BATCH_SIZE \
            --context-length $CONTEXT_LENGTH \
            --num-workers $NUM_WORKERS
    fi

elif [ "$MODE" == "precond" ]; then
    # Preconditioned space evaluation
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: MODEL_PATH must be set for precond mode"
        exit 1
    fi

    python comprehensive_evaluation_parallel.py \
        --mode precond \
        --model-path "$MODEL_PATH" \
        --precond-type $PRECOND_TYPE \
        --precond-degree $PRECOND_DEGREE \
        --batch-size $BATCH_SIZE \
        --context-length $CONTEXT_LENGTH \
        --num-workers $NUM_WORKERS

elif [ "$MODE" == "baseline-precond" ]; then
    # Baseline in preconditioned space evaluation
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: MODEL_PATH must be set for baseline-precond mode"
        exit 1
    fi

    python comprehensive_evaluation_parallel.py \
        --mode baseline-precond \
        --model-path "$MODEL_PATH" \
        --precond-type $PRECOND_TYPE \
        --precond-degree $PRECOND_DEGREE \
        --batch-size $BATCH_SIZE \
        --context-length $CONTEXT_LENGTH \
        --num-workers $NUM_WORKERS

else
    echo "ERROR: Invalid MODE=$MODE (must be standard, precond, or baseline-precond)"
    exit 1
fi

echo ""
echo "Evaluation completed!"
