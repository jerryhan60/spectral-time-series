#!/bin/bash
#SBATCH --job-name=m2_wt_base
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/m2_wt_base_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/m2_wt_base_%j.err

# Moirai2 Small baseline with WEIGHTED LOTSA (proportional sampling)
# 10K steps: 100 batches/epoch x 100 epochs, batch_size=256
# Purpose: Compare weighted vs unweighted LOTSA training data
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

echo "=== Moirai2 Small Weighted Baseline ==="
echo "Start: $(date) | Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Config: 10K steps, batch=256, data=lotsa_v1_weighted, warmup=10000"

python -m cli.train -cp conf/pretrain \
  run_name=m2_weighted_baseline \
  model=moirai2_small \
  model.log_on_step=true \
  model.anomaly_zscore_threshold=8.0 \
  model.anomaly_variance_ratio_threshold=0.0 \
  model.num_warmup_steps=10000 \
  data=lotsa_v1_weighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  trainer.enable_progress_bar=true \
  seed=42

echo "End: $(date)"
echo "Checkpoint: outputs/pretrain/moirai2_small/lotsa_v1_weighted/m2_weighted_baseline/checkpoints/epoch_99-step_10000.ckpt"
