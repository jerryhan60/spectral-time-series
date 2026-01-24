#!/bin/bash
#SBATCH --job-name=moirai_patched_precond
#SBATCH --output=logs/pretrain_patched_precond_%j.out
#SBATCH --error=logs/pretrain_patched_precond_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=spectralssmtorch
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jh1161@princeton.edu

# ============================================
# Moirai Pretraining with PATCH-LEVEL Preconditioning
# ============================================
#
# This script trains Moirai with patch-level preconditioning,
# where preconditioning is applied AFTER patching (to patches
# rather than individual time steps).
#
# Key differences from time-step level preconditioning:
# - Uses MoiraiPretrainPatched instead of MoiraiPretrain
# - Preconditioning formula applied across patch dimension:
#   ỹ[patch_t] = y[patch_t] + Σᵢ cᵢ · y[patch_t-i]
# - Loss is computed in preconditioned space
# - Does NOT support learnable coefficients or loss in original space
#
# Configurable via environment variables:
#   PRECOND_DEGREE: Polynomial degree (default: 5)
#   PRECOND_TYPE: chebyshev or legendre (default: chebyshev)
#
# Usage:
#   sbatch pretrain_moirai_patched_precond.slurm
#   sbatch --export=PRECOND_DEGREE=4 pretrain_moirai_patched_precond.slurm
# ============================================

echo "=========================================="
echo "Moirai Pretraining: PATCH-LEVEL Preconditioning"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Configuration from environment or defaults
PRECOND_DEGREE=${PRECOND_DEGREE:-5}
PRECOND_TYPE=${PRECOND_TYPE:-chebyshev}

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

# Generate run name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="patched_precond_${PRECOND_TYPE}_d${PRECOND_DEGREE}_${TIMESTAMP}"

echo "=========================================="
echo "Configuration:"
echo "  - Model: moirai_small_patched_precond"
echo "  - Preconditioning: PATCH-LEVEL (applied after patching)"
echo "  - Polynomial Type: ${PRECOND_TYPE}"
echo "  - Degree: ${PRECOND_DEGREE}"
echo "  - Loss Space: Preconditioned (patches)"
echo "  - Learnable Coefficients: No"
echo "Run Name: $RUN_NAME"
echo "=========================================="
echo ""

echo "Starting pretraining..."

python -m cli.train \
  -cp conf/pretrain \
  run_name=$RUN_NAME \
  model=moirai_small_patched_precond \
  data=lotsa_v1_unweighted \
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
