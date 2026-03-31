#!/bin/bash
#SBATCH --job-name=abl_dup_r
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=3:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/abl_dup_r_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/abl_dup_r_%j.err

# RESUME: Ablation duplicate input — timed out at epoch 89 (9K steps)
# Resume from epoch_89 checkpoint to complete remaining ~1K steps
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | RESUME: Ablation duplicate input"
python -m cli.train -cp conf/pretrain \
  run_name=ablation_duplicate_10k \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_ablation=duplicate \
  ckpt_path=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ablation_duplicate_10k/checkpoints/epoch_89-step_9000.ckpt
echo "Training done: $(date)"

# Auto-eval after training completes
cd /scratch/gpfs/EHAZAN/jh1161/gifteval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ablation_duplicate_10k/checkpoints/epoch_99-step_10000.ckpt"
if [ -f "$CKPT" ]; then
  echo "--- Evaluating abl_dup @ 10K ---"
  python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
else
  echo "WARNING: 10K checkpoint not found after resume"
fi
echo "End: $(date)"
