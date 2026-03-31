#!/bin/bash
#SBATCH --job-name=leg46_r
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=24:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/leg46_r_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/leg46_r_%j.err

# RESUME: Legendre d=4+d=6 + 10% dropout 100K — timed out at epoch 610 (ckpt: 60K)
# Resume from epoch_599 checkpoint to reach full 100K steps
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

echo "Start: $(date) | Node: $(hostname) | RESUME: Legendre d=4+d=6 + 10% dropout, 100K"
python -m cli.train -cp conf/pretrain \
  run_name=leg_ms46_100k \
  model=moirai2_small \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=legendre \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  +model.module_kwargs.time_precondition_extra_hints="6:16" \
  model.module_kwargs.hint_dropout=0.1 \
  model.anomaly_zscore_threshold=8.0 \
  model.num_warmup_steps=1000 \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=1000 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  trainer.enable_progress_bar=true \
  seed=42 \
  ckpt_path=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/leg_ms46_100k/checkpoints/epoch_599-step_60000.ckpt
echo "Training done: $(date)"

# Auto-eval
cd /scratch/gpfs/EHAZAN/jh1161
export HF_HUB_OFFLINE=1
CKPT="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/leg_ms46_100k/checkpoints/epoch_999-step_100000.ckpt"
if [ -f "$CKPT" ]; then
  echo "--- Evaluating leg46_100k @ 100K ---"
  python gifteval/eval_gifteval.py --checkpoint "$CKPT" --dataset-timeout 600
else
  echo "WARNING: 100K checkpoint not found, trying latest:"
  CKPT=$(ls -t uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/leg_ms46_100k/checkpoints/*.ckpt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    python gifteval/eval_gifteval.py --checkpoint "$CKPT" --dataset-timeout 600
  else
    echo "ERROR: No checkpoints found"
  fi
fi
echo "Done: $(date)"
