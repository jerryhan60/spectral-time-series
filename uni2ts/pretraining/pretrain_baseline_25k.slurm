#!/bin/bash
#SBATCH --job-name=m2_base_25k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=logs/m2_base_25k_%j.out
#SBATCH --error=logs/m2_base_25k_%j.err

# Baseline (no preconditioning) - 25K steps
# 250 epochs x 100 batches = 25K steps, bs=256

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

export HYDRA_FULL_ERROR=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Start: $(date) | Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "25K steps, bs=256, lr=1e-3, 2500 warmup, cosine"

python -m cli.train -cp conf/pretrain \
  run_name=m2_baseline_25k_${TIMESTAMP} \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=250 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 \
  train_dataloader.persistent_workers=true \
  model.num_warmup_steps=2500 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0

echo "Baseline complete (exit: $?): $(date)"
echo "Run: m2_baseline_25k_${TIMESTAMP}"
