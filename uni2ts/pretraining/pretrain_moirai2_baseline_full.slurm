#!/bin/bash
#SBATCH --job-name=m2_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=logs/m2_baseline_%j.out
#SBATCH --error=logs/m2_baseline_%j.err

# Moirai2 Small baseline (aligned with AK's config)
# 100K steps: 1000 epochs x 100 batches, bs=256, 10K warmup
# Key settings matched to AK: weighted data, zscore=6.0, var_ratio=0.0

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

export HYDRA_FULL_ERROR=1

echo "=== Moirai2 Baseline (Aligned with AK) ==="
echo "Start: $(date) | Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "100K steps, bs=256, lr=1e-3, 10K warmup, cosine, weighted data, zscore=6.0"
echo ""

python -m cli.train \
  -cp conf/pretrain \
  run_name=m2_baseline_$(date +%Y%m%d_%H%M%S) \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_moirai2 \
  trainer.max_epochs=1000 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 \
  train_dataloader.persistent_workers=true \
  model.num_warmup_steps=10000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=6.0 \
  model.anomaly_variance_ratio_threshold=0.0

echo ""
echo "=== Complete: $(date) ==="
