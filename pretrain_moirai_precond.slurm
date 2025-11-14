#!/bin/bash
#SBATCH --job-name=moirai_precond
#SBATCH --output=logs/pretrain_precond_%j.out
#SBATCH --error=logs/pretrain_precond_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jh1161@princeton.edu

# Print job information
echo "=========================================="
echo "Moirai Pretraining with Preconditioning"
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

# Preconditioning parameters (can be overridden via command line)
# Usage: sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=10 pretrain_moirai_precond.slurm
PRECOND_TYPE=${PRECOND_TYPE:-chebyshev}
PRECOND_DEGREE=${PRECOND_DEGREE:-5}
RUN_SUFFIX=${RUN_SUFFIX:-}

# Generate run name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="precond_${PRECOND_TYPE}_d${PRECOND_DEGREE}_${TIMESTAMP}${RUN_SUFFIX}"

echo "=========================================="
echo "Preconditioning Configuration:"
echo "=========================================="
echo "Polynomial Type: $PRECOND_TYPE"
echo "Degree: $PRECOND_DEGREE"
echo "Run Name: $RUN_NAME"
echo "=========================================="
echo ""

# Run pretraining with preconditioning
# Based on PRECONDITIONING_USAGE.md recommendations
echo "Starting pretraining with preconditioning..."

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
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
