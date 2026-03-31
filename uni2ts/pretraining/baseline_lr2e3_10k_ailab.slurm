#!/bin/bash
#SBATCH --job-name=bl_lr2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/bl_lr2_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/bl_lr2_%j.err

# Higher-LR baseline: Does BL with LR=2e-3 match HD10@LR=1e-3?
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"

echo "=== Baseline Higher-LR Sweep (ailab) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

echo -e "\n--- Training baseline_lr2e3_s0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain \
  run_name=baseline_lr2e3_s0_10k \
  $COMMON seed=0 \
  model.lr=2e-3

echo -e "\n--- Training baseline_lr3e3_s0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain \
  run_name=baseline_lr3e3_s0_10k \
  $COMMON seed=0 \
  model.lr=3e-3

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for RUN in baseline_lr2e3_s0_10k baseline_lr3e3_s0_10k; do
  ckpt="${BASE}/${RUN}/checkpoints/epoch_99-step_10000.ckpt"
  if [ -f "$ckpt" ]; then
    echo -e "\n--- Evaluating ${RUN} ---"
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
      --checkpoint "$ckpt" --context-length 4000 --batch-size 64
  else
    echo "SKIP: $ckpt not found"
  fi
done

echo "=== Done: $(date) ==="
