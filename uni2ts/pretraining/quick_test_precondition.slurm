#!/bin/bash
#SBATCH --job-name=m2_quick_precond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=slurm-m2-quick-precond-%j.out
#SBATCH --error=slurm-m2-quick-precond-%j.err

# Quick Moirai2 pretraining test WITH preconditioning
# Uses ak8836's best config: moirai2_regular3_z8_fir64_lambda01
#   - Time preconditioning: Chebyshev degree 5, stride 1
#   - FIR inverse: length 64, stride 1, lambda 0.1
#   - Anomaly z-score threshold: 8.0
#   - Coeffs init from regularization=3 run
# Uses test_small data (single dataset: australian_electricity_demand)
# 20 epochs x 10 batches = 200 training steps

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

export HYDRA_FULL_ERROR=1

python -m cli.train \
  -cp conf/pretrain \
  run_name=quick_precond_$(date +%Y%m%d_%H%M%S) \
  model=moirai2_small \
  data=test_small \
  trainer.max_epochs=20 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=10 \
  train_dataloader.batch_size=32 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=50 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.anomaly_variance_ratio_threshold=0.0 \
  model.time_precondition_reverse_in_loss=false \
  model.time_precondition_inverse_lambda=0.1 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=5 \
  model.module_kwargs.time_precondition_stride=1 \
  'model.module_kwargs.time_precondition_coeffs_init=[0.0,-0.11758551,0.0,-0.13611055]' \
  model.module_kwargs.time_precondition_inverse_enabled=true \
  model.module_kwargs.time_precondition_inverse_length=64 \
  model.module_kwargs.time_precondition_inverse_stride=1
