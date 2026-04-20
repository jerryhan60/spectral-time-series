#!/bin/bash
#SBATCH --job-name=hd100_sN
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=36:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/hd100_s%a_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/hd100_s%a_%j.err
#SBATCH --array=7,2

# Multi-seed HD10 100K + matched baseline
# Usage: sbatch --partition=pli --account=eladgroup --qos=pli-low hd10_100k_seeds.slurm
# Each array task runs one seed: BL 100K + HD10 100K (~30h total)

SEED=$SLURM_ARRAY_TASK_ID

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=1000 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 seed=${SEED}"
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.time_precondition_degree=4 model.module_kwargs.hint_dropout=0.1"

echo "=== HD10 100K multi-seed: seed=${SEED} ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Baseline 100K
echo -e "\n--- baseline_m2d_s${SEED}_100k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=baseline_m2d_s${SEED}_100k $COMMON

# 2. HD10 100K
echo -e "\n--- hd10_m2d_s${SEED}_100k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_m2d_s${SEED}_100k $COMMON $HINT

echo "=== Seed ${SEED} complete: $(date) ==="
