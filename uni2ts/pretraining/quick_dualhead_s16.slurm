#!/bin/bash
#SBATCH --job-name=q_dual_s16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=logs/q_dual_s16_%j.out
#SBATCH --error=logs/q_dual_s16_%j.err

# Dual-head mode: model sees preconditioned input, has TWO output heads.
# - Primary head (out_proj): predicts preconditioned targets
# - Raw head (out_proj_raw): predicts raw-space targets
# Training: raw_loss + lambda * precond_loss
# Inference: uses raw head only — NO reversal needed.
# Can fine-tune from baseline (strict=False, out_proj_raw is new).

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

export HYDRA_FULL_ERROR=1

echo "Start: $(date) | Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "EXPERIMENT: Dual-head mode — precond aux loss, raw primary (stride=16, d=4)"

python -m cli.train \
  -cp conf/pretrain \
  run_name=m2_dualhead_s16_$(date +%Y%m%d_%H%M%S) \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.anomaly_variance_ratio_threshold=0.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=5 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_dual_head=true \
  model.time_precondition_dual_head_lambda=1.0

echo "End: $(date)"
