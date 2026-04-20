#!/bin/bash
#SBATCH --job-name=core_m2d
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=18:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/core_m2d_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/core_m2d_%j.err

# Core experiments on lotsa_v1_moirai2 (correct Moirai2 data config)
# Priority: baseline, hd10, ms46_hd20 — all seed=0, 10K steps
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT_BASE="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.time_precondition_degree=4"

echo "=== Core M2D Experiments (seed=0) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Baseline (no hints)
echo -e "\n--- baseline_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=baseline_m2d_seed0_10k $COMMON

# 2. HD10: Chebyshev d=4, hint_dropout=0.1
echo -e "\n--- hd10_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.hint_dropout=0.1

# 3. MS46+HD20: Multi-scale d=4+d=6, hint_dropout=0.2
echo -e "\n--- ms46hd20_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=ms46hd20_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.hint_dropout=0.2 \
  +model.module_kwargs.time_precondition_extra_hints="6:16"

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for MODEL in baseline_m2d_seed0_10k hd10_m2d_seed0_10k ms46hd20_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== All core m2d runs complete: $(date) ==="
