#!/bin/bash
#SBATCH --job-name=deg_ema
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=18:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/deg_ema_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/deg_ema_%j.err

# Ablation experiments on m2d data, seed 0:
# 1. d=6 single-scale + 10% dropout (complete degree sweep: d=2,3,4,6)
# 2. Duplicate channel + 10% dropout (isolate polynomial vs extra capacity)
# 3. EMA hints + 10% dropout (alternative filter type)
# 4. d=8 single-scale + 10% dropout (if d=6 > d=4, does d=8 continue?)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT_BASE="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true"

echo "=== Degree & EMA Ablation (m2d, seed=0) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. d=6 single-scale + 10% dropout
echo -e "\n--- hd6_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd6_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_degree=6 \
  model.module_kwargs.hint_dropout=0.1

# 2. Duplicate channel + 10% dropout
echo -e "\n--- dup_hd10_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=dup_hd10_m2d_seed0_10k $COMMON \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_ablation=duplicate \
  model.module_kwargs.hint_dropout=0.1

# 3. EMA hints + 10% dropout
echo -e "\n--- ema_hint_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=ema_hint_m2d_seed0_10k $COMMON \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=ema \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1

# 4. d=8 single-scale + 10% dropout
echo -e "\n--- hd8_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd8_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_degree=8 \
  model.module_kwargs.hint_dropout=0.1

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

# Evaluate all 4
BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for MODEL in hd6_m2d_seed0_10k dup_hd10_m2d_seed0_10k ema_hint_m2d_seed0_10k hd8_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== All ablation runs complete: $(date) ==="
