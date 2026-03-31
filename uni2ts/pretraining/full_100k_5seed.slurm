#!/bin/bash
#SBATCH --job-name=f100k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=36:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/f100k_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/f100k_%A_%a.err
#SBATCH --array=0-7

# 100K training for seeds 1 and 42 (seeds 0,2,7 already done)
# Array 0-3: BL seeds 1,42 + HD10 seeds 1,42
# Array 4-7: Dup seeds 1,42 + Zero seeds 1,42
#
# Matches existing 100K config EXACTLY.
# Run: sbatch --partition=pli --account=eladgroup --qos=pli-low full_100k_5seed.slurm

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

SEEDS=(1 42 1 42 1 42 1 42)
MODELS=(bl bl hd10 hd10 dup dup zero zero)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 \
  trainer.max_epochs=1000 trainer.precision=bf16-mixed tf32=false \
  train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 \
  model.anomaly_variance_ratio_threshold=0.0 seed=${SEED}"

HINT_BASE="model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.hint_dropout=0.1"

case $MODEL in
  bl)
    RUN_NAME="baseline_m2d_s${SEED}_100k"
    HINT=""
    ;;
  hd10)
    RUN_NAME="hd10_m2d_s${SEED}_100k"
    HINT="$HINT_BASE"
    ;;
  dup)
    RUN_NAME="duplicate_m2d_s${SEED}_100k"
    HINT="$HINT_BASE model.module_kwargs.hint_ablation=duplicate"
    ;;
  zero)
    RUN_NAME="zero_m2d_s${SEED}_100k"
    HINT="$HINT_BASE model.module_kwargs.hint_ablation=zero"
    ;;
esac

echo "=== 100K: ${MODEL} seed ${SEED} | Run: ${RUN_NAME} ==="
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | Time: $(date)"

# Skip if checkpoint already exists
CKPT_PATH="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/${RUN_NAME}/checkpoints/epoch_999-step_100000.ckpt"
if [ -f "$CKPT_PATH" ]; then
  echo "Checkpoint exists, skipping training: $CKPT_PATH"
else
  python -m cli.train -cp conf/pretrain run_name=${RUN_NAME} $COMMON $HINT
fi

# Auto-eval
cd /scratch/gpfs/EHAZAN/jh1161
CKPT="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/${RUN_NAME}/checkpoints/epoch_999-step_100000.ckpt"
if [ -f "$CKPT" ]; then
  echo -e "\n--- eval ${RUN_NAME} ---\nTime: $(date)"
  python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
else
  echo "ERROR: Checkpoint not found at $CKPT"
fi

echo "=== ${MODEL} 100K seed ${SEED} complete: $(date) ==="
