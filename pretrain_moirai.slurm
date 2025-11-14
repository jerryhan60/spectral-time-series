#!/bin/bash
#SBATCH --job-name=moirai_pretrain
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
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

# Run pretraining
# Based on README: https://github.com/SalesforceAIResearch/uni2ts
# Command: python -m cli.train -cp conf/pretrain run_name=first_run model=moirai_small data=lotsa_v1_unweighted

python -m cli.train \
  -cp conf/pretrain \
  run_name=pretrain_run_$(date +%Y%m%d_%H%M%S) \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  seed=0

echo "End time: $(date)"
