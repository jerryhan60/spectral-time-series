#!/bin/bash
#SBATCH --job-name=q_hint_s16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=logs/q_hint_s16_%j.out
#SBATCH --error=logs/q_hint_s16_%j.err

# Spectral Hint mode: FIR filter residual as input feature.
# Model sees [target, mask, precond_residual] (3 channels per timepoint).
# Predicts in RAW space — NO reversal needed at inference.
# The hint tells the model about spectral structure without modifying targets.

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

export HYDRA_FULL_ERROR=1

echo "Start: $(date) | Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "EXPERIMENT: Spectral hint mode — precond residual as input feature (stride=16, d=4)"

python -m cli.train \
  -cp conf/pretrain \
  run_name=m2_hint_s16_$(date +%Y%m%d_%H%M%S) \
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
  model.module_kwargs.time_precondition_hint_mode=true

echo "End: $(date)"
