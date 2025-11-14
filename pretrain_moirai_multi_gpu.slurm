#!/bin/bash
#SBATCH --job-name=moirai_pretrain_multigpu
#SBATCH --output=logs/pretrain_multigpu_%j.out
#SBATCH --error=logs/pretrain_multigpu_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jh1161@princeton.edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Change to project directory
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Load environment
source venv/bin/activate

# Check GPU availability
nvidia-smi

# Run pretraining with multiple GPUs
# Uses PyTorch Lightning auto-detection for multi-GPU training

python -m cli.train \
  -cp conf/pretrain \
  run_name=pretrain_multigpu_$(date +%Y%m%d_%H%M%S) \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  trainer.devices=4 \
  trainer.strategy=ddp \
  seed=0

echo "End time: $(date)"
