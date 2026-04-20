#!/bin/bash
#SBATCH --job-name=abl100k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=36:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/abl100k_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/abl100k_%A_%a.err
#SBATCH --array=0-5

# 100K capacity controls: Duplicate + Zero, seeds 0,2,7
# Matches existing BL/HD10/MSHD10 100K runs EXACTLY (same config).
# Array 0-2: duplicate (seeds 0,2,7)
# Array 3-5: zero (seeds 0,2,7)
#
# Run: sbatch --partition=pli --account=eladgroup --qos=pli-low ablation_100k_3seed.slurm

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

SEEDS=(0 2 7 0 2 7)
ABLATIONS=(duplicate duplicate duplicate zero zero zero)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ABL=${ABLATIONS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="${ABL}_m2d_s${SEED}_100k"

# EXACT same config as hd10_m2d_s0_100k (verified from .hydra/config.yaml)
COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 \
  trainer.max_epochs=1000 trainer.precision=bf16-mixed tf32=false \
  train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 \
  model.anomaly_variance_ratio_threshold=0.0 seed=${SEED}"

# Hint config with ablation override — same arch as HD10 but different content
HINT="model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.hint_dropout=0.1 \
  model.module_kwargs.hint_ablation=${ABL}"

echo "=== 100K Ablation: ${ABL} | Seed: ${SEED} | Run: ${RUN_NAME} ==="
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | Time: $(date)"

python -m cli.train -cp conf/pretrain run_name=${RUN_NAME} $COMMON $HINT

# Auto-eval at 100K checkpoint
cd /scratch/gpfs/EHAZAN/jh1161
CKPT="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/${RUN_NAME}/checkpoints/epoch_999-step_100000.ckpt"
if [ -f "$CKPT" ]; then
  echo -e "\n--- eval ${RUN_NAME} ---\nTime: $(date)"
  python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
else
  echo "ERROR: Checkpoint not found at $CKPT"
fi

echo "=== ${ABL} 100K seed ${SEED} complete: $(date) ==="
