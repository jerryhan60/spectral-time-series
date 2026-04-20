#!/bin/bash
#SBATCH --job-name=ds_m2d
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ds_m2d_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ds_m2d_%j.err

# Dual-stride experiment: d=4@stride=16 + d=4@stride=8, with 10% hint dropout
# Motivation: HD10 (stride=16) is good at H/15T/W but weak at 5T/10S.
# Adding stride=8 covers shorter lookback distances for sub-hourly data.
# 2 hint channels (same degree, different strides).
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
# Primary: d=4, stride=16. Extra: d=4, stride=8.
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.hint_dropout=0.1 +model.module_kwargs.time_precondition_extra_hints=4:8"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== Dual-Stride d=4@s16 + d=4@s8 (m2d) ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for SEED in 0 1; do
  echo -e "\n--- ds48_hd10_m2d_seed${SEED}_10k ---\nTime: $(date)"
  python -m cli.train -cp conf/pretrain run_name=ds48_hd10_m2d_seed${SEED}_10k $COMMON seed=$SEED $HINT
done

echo "=== Training complete, starting eval ==="

for SEED in 0 1; do
  MODEL="ds48_hd10_m2d_seed${SEED}_10k"
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---\nTime: $(date)"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== Dual-stride complete: $(date) ==="
