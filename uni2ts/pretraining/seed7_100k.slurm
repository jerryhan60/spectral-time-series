#!/bin/bash
#SBATCH --job-name=s7_100k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=48:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/s7_100k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/s7_100k_%j.err

# 100K seed=7: ms46+drop and baseline
# Matches original 100K configs (10K warmup) but with seed=7 instead of 42
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

echo "=== 100K Seed=7 ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

echo "--- Training ms46+drop 100K seed=7 ---"
echo "Time: $(date)"
python -m cli.train -cp conf/pretrain \
  run_name=ms46_drop_seed7_100k \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=1000 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 \
  model.num_warmup_steps=10000 \
  trainer.enable_progress_bar=true \
  seed=7 \
  model.anomaly_zscore_threshold=8.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1 \
  +model.module_kwargs.time_precondition_extra_hints="6:16"

echo ""
echo "--- Training baseline 100K seed=7 ---"
echo "Time: $(date)"
python -m cli.train -cp conf/pretrain \
  run_name=baseline_seed7_100k \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=1000 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 \
  model.num_warmup_steps=10000 \
  trainer.enable_progress_bar=true \
  seed=7 \
  model.anomaly_zscore_threshold=8.0

echo "Seed=7 100K complete: $(date)"
