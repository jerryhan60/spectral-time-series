#!/bin/bash
#SBATCH --job-name=m2_s16_ws
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=logs/m2_s16_ws_%j.out
#SBATCH --error=logs/m2_s16_ws_%j.err

# Stride=16 d=4 with warm-start coefficients from prior run
# Init from learned values [-0.115, -0.039, -0.031, -0.001]
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
echo "25K steps, stride=16, d=4, warm-start from learned coeffs"

python -m cli.train -cp conf/pretrain \
  run_name=m2_stride16_warmstart_${TIMESTAMP} \
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
  model.anomaly_zscore_threshold=8.0 \
  model.time_precondition_inverse_lambda=0.0 \
  model.time_precondition_coeffs_lambda=0.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_learnable=true \
  model.module_kwargs.time_precondition_type=lyapunov \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_reg_lambda=5.0 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_inverse_enabled=false \
  'model.module_kwargs.time_precondition_coeffs_init=[-0.115,-0.039,-0.031,-0.001]'

echo "Complete (exit: $?): $(date)"
