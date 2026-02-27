#!/bin/bash
#SBATCH --job-name=ms46_100k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=36:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ms46_100k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ms46_100k_%j.err

# Multi-scale d=4+d=6 + 10% dropout, 100K steps
# Combines: best 10K config (ms d=4+d=6, -6.01%) + best 100K technique (dropout, -7.45%)
# This is the highest-priority experiment from experiment_summary.md
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: ms d=4+d=6 + 10% dropout, 100K steps"
python -m cli.train -cp conf/pretrain \
  run_name=ms_d4d6_100k_drop_$(date +%Y%m%d_%H%M%S) \
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
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1 \
  +model.module_kwargs.time_precondition_extra_hints="6:16"
echo "End: $(date)"
