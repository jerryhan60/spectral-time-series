#!/bin/bash
#SBATCH --job-name=moirai_pretrain_learnable
#SBATCH --output=logs/pretrain_learnable_%j.out
#SBATCH --error=logs/pretrain_learnable_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jh1161@princeton.edu

# Configurable preconditioning parameters (can override via --export)
export PRECOND_DEGREE=${PRECOND_DEGREE:-5}
export PRECOND_TYPE=${PRECOND_TYPE:-"chebyshev"}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Preconditioning: type=$PRECOND_TYPE, degree=$PRECOND_DEGREE, learnable=true"

# Change to project directory
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Load environment
source venv/bin/activate

# Check GPU availability
nvidia-smi

# Run pretraining with learnable preconditioning
# Uses moirai_small_precond config (inherits from moirai_small)
# Enables learnable_preconditioning and loss_in_original_space
# This allows the model to learn optimal coefficients during training

python -m cli.train \
  -cp conf/pretrain \
  run_name=pretrain_learnable_${PRECOND_TYPE}_d${PRECOND_DEGREE}_$(date +%Y%m%d_%H%M%S) \
  model=moirai_small_precond \
  model.precondition_type=$PRECOND_TYPE \
  model.precondition_degree=$PRECOND_DEGREE \
  model.learnable_preconditioning=true \
  model.loss_in_original_space=true \
  data=lotsa_v1_unweighted \
  seed=0

echo "End time: $(date)"
