#!/bin/bash
#SBATCH --job-name=deg57
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/deg57_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/deg57_%A_%a.err
#SBATCH --array=0-1

# Missing degree ablation: d=5 and d=7 (seed 0, 10% dropout, s=16)

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

DEGREES=(5 7)
DEG=${DEGREES[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="hd${DEG}_m2d_seed0_10k"

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 \
  trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false \
  train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 \
  model.anomaly_variance_ratio_threshold=0.0 seed=0"

HINT="model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_degree=${DEG} \
  model.module_kwargs.hint_dropout=0.1"

echo "=== Degree ${DEG}: ${RUN_NAME} ==="
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | Time: $(date)"

python -m cli.train -cp conf/pretrain run_name=${RUN_NAME} $COMMON $HINT

cd /scratch/gpfs/EHAZAN/jh1161
CKPT="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/${RUN_NAME}/checkpoints/epoch_99-step_10000.ckpt"
if [ -f "$CKPT" ]; then
  echo -e "\n--- eval ${RUN_NAME} ---\nTime: $(date)"
  python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
fi
echo "=== d=${DEG} complete: $(date) ==="
