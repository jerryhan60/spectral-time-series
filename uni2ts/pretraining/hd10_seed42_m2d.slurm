#!/bin/bash
#SBATCH --job-name=hd10_s42
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/hd10_s42_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/hd10_s42_%j.err

# HD10 seed 42 on m2d data — fills gap for 5-seed HD10 coverage
# (seeds 0,1,2 done, seed 7 done, seed 42 missing)
# Also trains baseline seed 42 for paired comparison
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.hint_dropout=0.1"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== HD10 + BL seed 42 (m2d) ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

# Train baseline seed 42
echo -e "\n--- baseline_m2d_seed42_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=baseline_m2d_seed42_10k $COMMON seed=42

# Train HD10 seed 42
echo -e "\n--- hd10_m2d_seed42_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_m2d_seed42_10k $COMMON seed=42 $HINT

echo "=== Training complete, starting eval ==="

for MODEL in baseline_m2d_seed42_10k hd10_m2d_seed42_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---\nTime: $(date)"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== HD10 seed 42 complete: $(date) ==="
