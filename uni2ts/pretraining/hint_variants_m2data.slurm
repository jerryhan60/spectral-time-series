#!/bin/bash
#SBATCH --job-name=hint_var
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/hint_var_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/hint_var_%j.err

# Hint variant ablation on lotsa_v1_moirai2 (seed=0 only, quick screen)
# Tests: ms46, hd10_drop20, hd10_d2, hd10_d3
# If any look promising, follow up with 3-seed runs
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT_BASE="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true"

echo "=== Hint Variant Ablation (lotsa_v1_moirai2, seed=0) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. MS46 (multi-scale d=4+d=6, no dropout)
echo -e "\n--- ms46_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=ms46_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_degree=4 \
  +model.module_kwargs.time_precondition_extra_hints="6:16"

# 2. hd10 with 20% dropout
echo -e "\n--- hd10_drop20_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_drop20_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.hint_dropout=0.2

# 3. d=2 hint + 10% dropout (less aggressive filtering)
echo -e "\n--- hd2_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd2_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_degree=2 \
  model.module_kwargs.hint_dropout=0.1

# 4. d=3 hint + 10% dropout
echo -e "\n--- hd3_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd3_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_degree=3 \
  model.module_kwargs.hint_dropout=0.1

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for MODEL in ms46_m2d_seed0_10k hd10_drop20_m2d_seed0_10k hd2_m2d_seed0_10k hd3_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== All variant runs complete: $(date) ==="
