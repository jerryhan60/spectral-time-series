#!/bin/bash
#SBATCH --job-name=ms10_100k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=36:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ms10_100k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ms10_100k_%j.err

# MSHD10 100K on moirai2 data (seed 0) + baseline 100K for comparison
# Highest-priority experiment: combines best method (MSHD10) + best data (moirai2) + long training
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=1000 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.time_precondition_degree=4 model.module_kwargs.hint_dropout=0.1 +model.module_kwargs.time_precondition_extra_hints=6:16"

echo "=== MSHD10 100K on moirai2 data ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Baseline 100K (moirai2 data, seed 0, 1K warmup)
echo -e "\n--- baseline_m2d_s0_100k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=baseline_m2d_s0_100k $COMMON

# 2. MSHD10 100K (moirai2 data, seed 0, 1K warmup)
echo -e "\n--- mshd10_m2d_s0_100k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=mshd10_m2d_s0_100k $COMMON $HINT

echo "=== Done: $(date) ==="
