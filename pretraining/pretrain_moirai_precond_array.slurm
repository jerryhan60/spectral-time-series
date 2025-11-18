#!/bin/bash
#SBATCH --job-name=precond_sweep
#SBATCH --output=logs/pretrain_precond_d%a_%A.out
#SBATCH --error=logs/pretrain_precond_d%a_%A.err
#SBATCH --array=1-10
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=hazan_intern
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=jh1161@princeton.edu

# =============================================================================
# Parallel Pretraining with Chebyshev Preconditioning Sweep
# =============================================================================
# This SLURM array job runs pretraining for Chebyshev degrees 1-10 in parallel
# Each array task (1-10) corresponds to one polynomial degree
#
# Usage:
#   sbatch pretraining/pretrain_moirai_precond_array.slurm
#
# The array index $SLURM_ARRAY_TASK_ID determines the polynomial degree
# =============================================================================

# The array task ID corresponds directly to the polynomial degree
PRECOND_DEGREE=$SLURM_ARRAY_TASK_ID
PRECOND_TYPE="chebyshev"

# Print job information
echo "=========================================="
echo "Moirai Pretraining - Chebyshev Sweep"
echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""
echo "Preconditioning Configuration:"
echo "  Polynomial Type: $PRECOND_TYPE"
echo "  Degree: $PRECOND_DEGREE"
echo "=========================================="
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

# Generate run name with timestamp and array task ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="precond_${PRECOND_TYPE}_d${PRECOND_DEGREE}_${TIMESTAMP}_task${SLURM_ARRAY_TASK_ID}"

echo "Run Name: $RUN_NAME"
echo ""
echo "Starting pretraining..."
echo "=========================================="
echo ""

# Run pretraining with preconditioning
python -m cli.train \
  -cp conf/pretrain \
  run_name=$RUN_NAME \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  model.enable_preconditioning=true \
  model.precondition_type=$PRECOND_TYPE \
  model.precondition_degree=$PRECOND_DEGREE \
  seed=0

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID (Degree $PRECOND_DEGREE) completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
