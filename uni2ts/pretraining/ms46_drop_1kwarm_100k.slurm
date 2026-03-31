#!/bin/bash
#SBATCH --job-name=ms46d1k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=36:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ms46d1k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ms46d1k_%j.err

# MS46 + 10% dropout + 1K warmup, 100K steps
# Critical experiment: combines best filter (ms46) + best dropout (10%) + best warmup (1K)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: ms46+drop10+1Kwarm 100K"

python -m cli.train -cp conf/pretrain \
  run_name=ms46_drop_1kwarm_100k_$(date +%Y%m%d_%H%M%S) \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=1000 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 \
  model.lr=1e-3 \
  model.module_kwargs.time_precondition_type=chebyshev \
  "+model.module_kwargs.time_precondition_extra_hints=6:16" \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0

echo "Training done: $(date)"

# Auto-evaluate at 100K
CKPT=$(ls -t outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms46_drop_1kwarm_100k_*/checkpoints/epoch_999-step_100000.ckpt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    echo "--- Evaluating ms46+drop+1kwarm @ 100K ---"
    cd /scratch/gpfs/EHAZAN/jh1161/gifteval
    python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
fi

echo "All done: $(date)"
