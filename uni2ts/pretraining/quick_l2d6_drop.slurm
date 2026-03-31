#!/bin/bash
#SBATCH --job-name=l2d6d_46
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/l2d6d_46_%j.out

module load anaconda3/2024.6
module load cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: L2-opt d=6 + 10% hint dropout, 10K"
echo "Hypothesis: L2-opt d=6 (1.1784) + dropout may match or beat ms46 (1.1675)"

python -m cli.train -cp conf/pretrain \
  run_name=l2d6_drop10_10k \
  model=moirai2_small \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=l2opt \
  model.module_kwargs.time_precondition_degree=6 \
  model.module_kwargs.time_precondition_stride=16 \
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
export HF_HUB_OFFLINE=1
CKPT=$(ls -t uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/l2d6_drop10_10k/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
  python gifteval/eval_gifteval.py --checkpoint "$CKPT" --dataset-timeout 600
fi
echo "Done: $(date)"
