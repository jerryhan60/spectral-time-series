#!/bin/bash
#SBATCH --job-name=moirai2_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=logs/moirai2_small_%j.out
#SBATCH --error=logs/moirai2_small_%j.err

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

echo "=== MOIRAI 2.0 Small Pretraining ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "LOTSA_V1_PATH=$LOTSA_V1_PATH"
echo ""
echo "Config: 10K steps (100 batches/epoch x 100 epochs), batch_size=128, bf16-mixed"
echo "Estimated time: ~2-3 hours on H200"
echo ""

# 10K steps: 100 batches/epoch x 100 epochs
# log_on_step=true to track per-step training loss
python -m cli.train \
  -cp conf/pretrain \
  run_name=moirai2_small_v1_$(date +%Y%m%d_%H%M%S) \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=128 \
  train_dataloader.num_workers=11 \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42

echo ""
echo "=== Pretraining Complete ==="
echo "End time: $(date)"
