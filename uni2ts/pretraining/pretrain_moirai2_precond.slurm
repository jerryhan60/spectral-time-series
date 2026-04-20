#!/bin/bash
#SBATCH --job-name=m2_precond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=logs/m2_precond_d${DEGREE}_%j.out
#SBATCH --error=logs/m2_precond_d${DEGREE}_%j.err

# Moirai2 Small with Chebyshev preconditioning
# 100K steps: 1000 epochs x 100 batches, bs=256, 10K warmup
# Pure Chebyshev coefficients (computed from degree), FIR inverse enabled
#
# Usage: sbatch --export=DEGREE=5 pretrain_moirai2_precond.slurm
# Or with custom coeffs: sbatch --export=DEGREE=5,COEFFS="[0.0,-0.1176,0.0,-0.1361]" pretrain_moirai2_precond.slurm

if [ -z "$DEGREE" ]; then
    echo "ERROR: DEGREE not set. Usage: sbatch --export=DEGREE=5 pretrain_moirai2_precond.slurm"
    exit 1
fi

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

export HYDRA_FULL_ERROR=1

# Build coeffs argument
COEFFS_ARG=""
if [ -n "$COEFFS" ]; then
    COEFFS_ARG="'model.module_kwargs.time_precondition_coeffs_init=${COEFFS}'"
    LABEL="custom"
else
    LABEL="cheb"
fi

echo "=== Moirai2 Precond (degree=${DEGREE}, ${LABEL}) ==="
echo "Start: $(date) | Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "100K steps, bs=256, Chebyshev deg=${DEGREE}, FIR inv len=64, lambda=0.1"
if [ -n "$COEFFS" ]; then
    echo "Custom coefficients: ${COEFFS}"
fi
echo ""

RUN_NAME="m2_precond_d${DEGREE}_${LABEL}_$(date +%Y%m%d_%H%M%S)"

if [ -n "$COEFFS" ]; then
    python -m cli.train \
      -cp conf/pretrain \
      run_name=${RUN_NAME} \
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
      seed=42 \
      model.anomaly_zscore_threshold=8.0 \
      model.time_precondition_reverse_in_loss=false \
      model.time_precondition_inverse_lambda=0.1 \
      model.module_kwargs.time_precondition_enabled=true \
      model.module_kwargs.time_precondition_type=chebyshev \
      model.module_kwargs.time_precondition_degree=${DEGREE} \
      model.module_kwargs.time_precondition_stride=1 \
      'model.module_kwargs.time_precondition_coeffs_init='"${COEFFS}" \
      model.module_kwargs.time_precondition_inverse_enabled=true \
      model.module_kwargs.time_precondition_inverse_length=64 \
      model.module_kwargs.time_precondition_inverse_stride=1
else
    python -m cli.train \
      -cp conf/pretrain \
      run_name=${RUN_NAME} \
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
      seed=42 \
      model.anomaly_zscore_threshold=8.0 \
      model.time_precondition_reverse_in_loss=false \
      model.time_precondition_inverse_lambda=0.1 \
      model.module_kwargs.time_precondition_enabled=true \
      model.module_kwargs.time_precondition_type=chebyshev \
      model.module_kwargs.time_precondition_degree=${DEGREE} \
      model.module_kwargs.time_precondition_stride=1 \
      model.module_kwargs.time_precondition_inverse_enabled=true \
      model.module_kwargs.time_precondition_inverse_length=64 \
      model.module_kwargs.time_precondition_inverse_stride=1
fi

echo ""
echo "=== Complete: $(date) ==="
