#!/bin/bash
#SBATCH --job-name=drop_var
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/drop_var_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/drop_var_%j.err

# Dropout variant ablation on lotsa_v1_moirai2 (seed=0 screen)
# Hypothesis: more aggressive dropout makes hints robust for short horizons
# Tests: hd30, sd30, hd10+sd20, ms46_hd20
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT_BASE="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.time_precondition_degree=4"

echo "=== Dropout Variant Ablation (lotsa_v1_moirai2, seed=0) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. hd30: 30% patch-level hint dropout (d=4)
echo -e "\n--- hd30_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd30_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.hint_dropout=0.3

# 2. sd30: 30% sequence-level hint dropout (d=4, no patch dropout)
echo -e "\n--- sd30_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=sd30_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.hint_sequence_dropout=0.3

# 3. hd10+sd20: combined dropout
echo -e "\n--- hd10sd20_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10sd20_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.hint_dropout=0.1 \
  model.module_kwargs.hint_sequence_dropout=0.2

# 4. ms46+hd20: multi-scale with stronger dropout
echo -e "\n--- ms46hd20_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=ms46hd20_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.hint_dropout=0.2 \
  +model.module_kwargs.time_precondition_extra_hints="6:16"

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for MODEL in hd30_m2d_seed0_10k sd30_m2d_seed0_10k hd10sd20_m2d_seed0_10k ms46hd20_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== All dropout variant runs complete: $(date) ==="
