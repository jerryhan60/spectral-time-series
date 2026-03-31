#!/bin/bash
#SBATCH --job-name=base_hd10
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=24:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/base_hd10_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/base_hd10_%j.err

# Model scaling experiment: HD10 vs baseline at BASE scale (d=768, L=12)
# Tests whether hint preconditioning benefit persists at larger model sizes
# Small: 11.4M params → Base: ~46M params (4x)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_base model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=64 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=42 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"

echo "=== Base Scale: HD10 vs Baseline (seed=42) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Baseline (base scale)
echo -e "\n--- baseline_base_m2d_seed42_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=baseline_base_m2d_seed42_10k $COMMON

# 2. HD10 (base scale)
echo -e "\n--- hd10_base_m2d_seed42_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_base_m2d_seed42_10k $COMMON \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_base/lotsa_v1_moirai2
for MODEL in baseline_base_m2d_seed42_10k hd10_base_m2d_seed42_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 32
done

echo "=== Base scale experiment complete: $(date) ==="
