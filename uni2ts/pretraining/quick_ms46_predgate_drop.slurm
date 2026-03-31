#!/bin/bash
#SBATCH --job-name=pgd46
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/pgd46_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/pgd46_%j.err

module load anaconda3/2024.6
module load cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: ms46 + pred gate + 10% dropout"

python -m cli.train -cp conf/pretrain \
  run_name=ms46_predgate_drop_10k \
  model=moirai2_small \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  +model.module_kwargs.time_precondition_extra_hints="6:16" \
  model.module_kwargs.hint_pred_gate=true \
  model.module_kwargs.hint_dropout=0.1 \
  model.anomaly_zscore_threshold=8.0 \
  model.num_warmup_steps=1000 \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  trainer.enable_progress_bar=true \
  seed=42

echo "Training complete: $(date)"

cd /scratch/gpfs/EHAZAN/jh1161
set -a; source uni2ts/.env; set +a
export HF_HUB_OFFLINE=1
CKPT=$(ls -t uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms46_predgate_drop_10k/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
  echo "=== Evaluating pred gate + drop model (ctx=4000) ==="
  python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000
fi
echo "Done: $(date)"
