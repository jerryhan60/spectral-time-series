#!/bin/bash
#SBATCH --job-name=moirai_precond_d4_orig_loss
#SBATCH --output=logs/pretrain_precond_d4_orig_loss_%j.out
#SBATCH --error=logs/pretrain_precond_d4_orig_loss_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=hazan_intern
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jh1161@princeton.edu

# Print job information
echo "=========================================="
echo "Moirai Pretraining: Degree 4, Loss in Original Space"
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

# Generate run name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="precond_d4_orig_loss_${TIMESTAMP}"

echo "=========================================="
echo "Configuration:"
echo "  - Model: moirai_small_precond"
echo "  - Polynomial Type: chebyshev"
echo "  - Degree: 4"
echo "  - Loss in Original Space: True"
echo "  - Learnable Preconditioning: False"
echo "Run Name: $RUN_NAME"
echo "=========================================="
echo ""

echo "Starting pretraining..."

python -m cli.train \
  -cp conf/pretrain \
  run_name=$RUN_NAME \
  model=moirai_small_precond \
  data=lotsa_v1_unweighted \
  model.precondition_degree=4 \
  model.loss_in_original_space=true \
  model.learnable_preconditioning=false \
  seed=0

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
