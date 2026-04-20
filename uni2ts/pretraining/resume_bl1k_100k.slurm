#!/bin/bash
#SBATCH --job-name=bl1k_r
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=8:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/bl1k_r_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/bl1k_r_%j.err

# RESUME: Baseline 1K warmup 100K — timed out at epoch 932 (latest ckpt: 90K)
# Resume from epoch_899 checkpoint to reach full 100K steps
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | RESUME: Baseline 1K warmup, 100K steps"
python -m cli.train -cp conf/pretrain \
  run_name=baseline_1kwarm_100k_20260228_063343 \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=1000 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  ckpt_path=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/baseline_1kwarm_100k_20260228_063343/checkpoints/epoch_899-step_90000.ckpt
echo "Training done: $(date)"

# Auto-eval after training completes
cd /scratch/gpfs/EHAZAN/jh1161/gifteval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/baseline_1kwarm_100k_20260228_063343/checkpoints/epoch_999-step_100000.ckpt"
if [ -f "$CKPT" ]; then
  echo "--- Evaluating bl1k_100k @ 100K ---"
  python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
else
  echo "WARNING: 100K checkpoint not found after resume"
fi
echo "End: $(date)"
