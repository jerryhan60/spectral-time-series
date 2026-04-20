#!/bin/bash
#SBATCH --job-name=q_dualloss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=logs/q_dualloss_%j.out
#SBATCH --error=logs/q_dualloss_%j.err

# Quick 10K: stride=16, d=4, DUAL LOSS
# Uses inverse_lambda to add original-space FIR loss as auxiliary term
# base_loss (preconditioned) + 0.5 * inverse_aux_loss (original space)
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Start: $(date) | Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "EXPERIMENT: Dual loss — preconditioned + original-space FIR inverse aux loss"

python -m cli.train -cp conf/pretrain \
  run_name=q_dualloss_${TIMESTAMP} \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 \
  train_dataloader.persistent_workers=true \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.time_precondition_reverse_in_loss=false \
  model.time_precondition_inverse_lambda=0.5 \
  model.time_precondition_coeffs_lambda=0.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_learnable=true \
  model.module_kwargs.time_precondition_type=lyapunov \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_reg_lambda=5.0 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_inverse_enabled=true \
  model.module_kwargs.time_precondition_inverse_length=64 \
  model.module_kwargs.time_precondition_inverse_stride=1

echo "Done (exit: $?): $(date)"
