#!/bin/bash
#SBATCH --job-name=str_swp
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=hazan_intern --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/str_swp_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/str_swp_%j.err

# Stride sweep: test strides 4, 8, 32 (we already have stride=16 as HD10)
# Also test multi-stride (4+8+16+32) = 16 hint channels
# All on m2d seed 0 for clean comparison with HD10 (-4.99%)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT_BASE="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.hint_dropout=0.1"

echo "=== Stride Sweep (m2d, seed=0) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. stride=4 (high-frequency bands)
echo -e "\n--- hd10_s4_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_s4_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_stride=4

# 2. stride=8 (intermediate)
echo -e "\n--- hd10_s8_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_s8_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_stride=8

# 3. stride=32 (low-frequency bands)
echo -e "\n--- hd10_s32_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_s32_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_stride=32

# 4. Multi-stride: primary stride=16, extra strides 4,8,32 (16 channels total)
echo -e "\n--- multi_stride_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=multi_stride_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_stride=16 \
  +model.module_kwargs.time_precondition_extra_hints="4:4|4:8|4:32"

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for MODEL in hd10_s4_m2d_seed0_10k hd10_s8_m2d_seed0_10k hd10_s32_m2d_seed0_10k multi_stride_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== Stride sweep complete: $(date) ==="
