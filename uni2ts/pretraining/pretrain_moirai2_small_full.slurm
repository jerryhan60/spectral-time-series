#!/bin/bash
#SBATCH --job-name=moirai2_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=logs/moirai2_small_full_%j.out
#SBATCH --error=logs/moirai2_small_full_%j.err

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

echo "=== MOIRAI 2.0 Small Full Pretraining (Paper Specs) ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "LOTSA_V1_PATH=$LOTSA_V1_PATH"
echo ""
echo "Config: 100K steps (100 batches/epoch x 1000 epochs), batch_size=256, bf16-mixed"
echo "Warmup: 10,000 steps (linear warmup then cosine annealing)"
echo "LR=1e-3, AdamW (beta1=0.9, beta2=0.98), weight_decay=1e-1"
echo ""

# 100K steps: 100 batches/epoch x 1000 epochs
# Paper specs: batch_size=256, 10K warmup, 100K total steps
python -m cli.train \
  -cp conf/pretrain \
  run_name=moirai2_small_full_$(date +%Y%m%d_%H%M%S) \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=1000 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=11 \
  model.num_warmup_steps=10000 \
  trainer.enable_progress_bar=true \
  seed=42

echo ""
echo "=== Pretraining Complete ==="
echo "End time: $(date)"
