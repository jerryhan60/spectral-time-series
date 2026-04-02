#!/bin/bash
#SBATCH --job-name=ema_ms
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=hazan_intern --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ema_ms_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ema_ms_%A_%a.err
#SBATCH --array=0-2

# Multi-seed EMA ablation: d=4, s=16, 10% dropout, seeds 1,2,7
# Seed 0 already exists (ema_hint_m2d_seed0_10k).

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

SEEDS=(1 2 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="ema_hint_m2d_seed${SEED}_10k"

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 \
  trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false \
  train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 \
  model.anomaly_variance_ratio_threshold=0.0 seed=${SEED}"

HINT="model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=ema \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1"

echo "=== EMA d=4 | Seed: ${SEED} | Run: ${RUN_NAME} ==="
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | Time: $(date)"

# Skip if checkpoint already exists
CKPT="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/${RUN_NAME}/checkpoints/epoch_99-step_10000.ckpt"
if [ -f "$CKPT" ]; then
  echo "Checkpoint already exists, skipping training: $CKPT"
else
  python -m cli.train -cp conf/pretrain run_name=${RUN_NAME} $COMMON $HINT
fi

# Auto-eval
cd /scratch/gpfs/EHAZAN/jh1161
CKPT="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/${RUN_NAME}/checkpoints/epoch_99-step_10000.ckpt"
if [ -f "$CKPT" ]; then
  echo -e "\n--- eval ${RUN_NAME} ---\nTime: $(date)"
  python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
else
  echo "ERROR: Checkpoint not found at $CKPT"
fi

echo "=== EMA seed ${SEED} complete: $(date) ==="
