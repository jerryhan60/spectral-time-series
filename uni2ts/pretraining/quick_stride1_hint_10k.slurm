#!/bin/bash
#SBATCH --job-name=s1_hint
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=hazan_intern
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/s1_hint_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/s1_hint_%j.err

# STRIDE ABLATION: Hint d=6 with stride=1 (dense FIR) vs stride=16 (patch-aligned)
# Tests the patch-alignment hypothesis: stride=16 = patch_size is critical
# Comparison points: s=16/d=6 → 1.1836, s=8/d=6 → ~1.2204, s=4/d=6 → 1.2346
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Hint d=6 stride=1 (dense FIR)"
python -m cli.train -cp conf/pretrain \
  run_name=hint_d6_s1_10k \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=6 \
  model.module_kwargs.time_precondition_stride=1 \
  model.module_kwargs.time_precondition_hint_mode=true
echo "Training done: $(date)"

# Auto-eval
cd /scratch/gpfs/EHAZAN/jh1161/gifteval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/hint_d6_s1_10k/checkpoints/epoch_99-step_10000.ckpt"
if [ -f "$CKPT" ]; then
  echo "--- Evaluating hint d=6 stride=1 @ 10K ---"
  python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
fi
echo "End: $(date)"
