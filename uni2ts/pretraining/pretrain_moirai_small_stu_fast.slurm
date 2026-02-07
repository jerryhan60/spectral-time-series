#!/bin/bash
#SBATCH --job-name=pretrain_stu_fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/uni2ts/logs/pretrain_stu_fast_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/uni2ts/logs/pretrain_stu_fast_%j.err

# STU-MOIRAI Pretraining with optimized forward_batched (2x faster)

# Load modules
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate virtual environment
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate

# Load environment variables
set -a
source .env
set +a

echo "=== STU-MOIRAI Fast Pretraining (forward_batched) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "Start time: $(date)"
echo "LOTSA_V1_PATH=$LOTSA_V1_PATH"
echo ""

# Run full pretraining with hybrid STU model
python -m cli.train \
  -cp conf/pretrain \
  run_name=moirai_small_stu_fast_$(date +%Y%m%d_%H%M%S) \
  model=moirai_small_stu \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=1000 \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=128 \
  trainer.enable_progress_bar=true \
  seed=42

echo ""
echo "=== Pretraining Complete ==="
echo "End time: $(date)"
