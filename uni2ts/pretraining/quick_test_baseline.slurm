#!/bin/bash
#SBATCH --job-name=m2_quick_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=slurm-m2-quick-base-%j.out
#SBATCH --error=slurm-m2-quick-base-%j.err

# Quick baseline Moirai2 pretraining test (no preconditioning)
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
  run_name=quick_baseline_$(date +%Y%m%d_%H%M%S) \
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
  seed=42
