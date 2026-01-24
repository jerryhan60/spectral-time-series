#!/bin/bash
#SBATCH --job-name=moirai_embed_precond
#SBATCH --output=logs/pretrain_embed_precond_%j.out
#SBATCH --error=logs/pretrain_embed_precond_%j.err
#SBATCH --time=16:00:00
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
echo "Moirai Pretraining with EMBEDDING-LEVEL Preconditioning"
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

# Embedding preconditioning parameters (can be overridden via command line)
# Usage: sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=3 pretraining/pretrain_moirai_embedding_precond.slurm
#
# Parameters:
#   PRECOND_TYPE: chebyshev (default) or legendre
#   PRECOND_DEGREE: polynomial degree (default 2, recommended 2-5)
#   NUM_TARGET_VARIATES: number of target variates to precondition (default: null = all)
#   RUN_SUFFIX: optional suffix for run name
#
PRECOND_TYPE=${PRECOND_TYPE:-chebyshev}
PRECOND_DEGREE=${PRECOND_DEGREE:-2}
NUM_TARGET_VARIATES=${NUM_TARGET_VARIATES:-null}
RUN_SUFFIX=${RUN_SUFFIX:-}

# Generate run name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="embed_precond_${PRECOND_TYPE}_d${PRECOND_DEGREE}_${TIMESTAMP}${RUN_SUFFIX}"

echo "=========================================="
echo "EMBEDDING-LEVEL Preconditioning Configuration:"
echo "=========================================="
echo "Polynomial Type: $PRECOND_TYPE"
echo "Degree: $PRECOND_DEGREE"
echo "Num Target Variates: $NUM_TARGET_VARIATES"
echo "Run Name: $RUN_NAME"
echo "=========================================="
echo ""
echo "NOTE: Embedding-level preconditioning applies polynomial convolution"
echo "to patch embeddings AFTER the projection layer, BEFORE the transformer."
echo ""
echo "Following Universal Sequence Preconditioning theory:"
echo "  - Only target variates (y_t) are preconditioned"
echo "  - Covariate variates (u_t) are left unchanged"
echo "  - Reversal after transformer is DISABLED (not a true inverse)"
echo ""

# Run pretraining with embedding-level preconditioning
echo "Starting pretraining with embedding-level preconditioning..."

python -m cli.train \
  -cp conf/pretrain \
  run_name=$RUN_NAME \
  model=moirai_small_embedding_precond \
  data=lotsa_v1_unweighted \
  model.module_kwargs.embedding_precondition_type=$PRECOND_TYPE \
  model.module_kwargs.embedding_precondition_degree=$PRECOND_DEGREE \
  model.module_kwargs.num_target_variates=$NUM_TARGET_VARIATES \
  seed=0

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
