#!/bin/bash
#SBATCH --job-name=pretrain_stu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=logs/pretrain_stu_%j.out
#SBATCH --error=logs/pretrain_stu_%j.err

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

echo "=== STU-MOIRAI Full Pretraining ==="
echo "Start time: $(date)"
echo "LOTSA_V1_PATH=$LOTSA_V1_PATH"

# Run full pretraining with hybrid STU model
python -m cli.train \
  -cp conf/pretrain \
  run_name=moirai_small_stu_$(date +%Y%m%d_%H%M%S) \
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
