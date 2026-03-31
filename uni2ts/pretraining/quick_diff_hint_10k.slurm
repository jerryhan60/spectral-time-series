#!/bin/bash
#SBATCH --job-name=diff_h10k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/diff_h_10k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/diff_h_10k_%j.err

# Ablation: Differencing hint (high-pass) instead of Chebyshev
# Tests whether simple differencing gives same benefit as polynomial structure
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

echo "Start: $(date) | Node: $(hostname) | Differencing hint d=6 s=16"

python -m cli.train -cp conf/pretrain \
  run_name=hint_diff_d6_10k \
  model=moirai2_small \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=differencing \
  model.module_kwargs.time_precondition_degree=6 \
  model.module_kwargs.time_precondition_stride=16 \
  model.anomaly_zscore_threshold=8.0 \
  model.num_warmup_steps=1000 \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  train_dataloader.num_batches_per_epoch=100 \
  trainer.accelerator=gpu trainer.devices=1

echo "Done: $(date)"
