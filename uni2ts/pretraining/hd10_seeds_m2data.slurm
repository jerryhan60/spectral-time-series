#!/bin/bash
#SBATCH --job-name=hd10_m2d
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=8:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/hd10_m2d_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/hd10_m2d_%j.err

# hd10 seed sensitivity with proper Moirai 2 data config (lotsa_v1_moirai2)
# Seeds 0,1,2 for hd10 (d=4, stride=16, dropout=0.1) + baseline
# Then auto-eval all 6 checkpoints
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

echo "=== hd10 + Baseline Seed Sensitivity (lotsa_v1_moirai2) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for SEED in 0 1 2; do
  echo ""
  echo "--- Training hd10 seed=$SEED (moirai2 data) ---"
  echo "Time: $(date)"
  python -m cli.train -cp conf/pretrain \
    run_name=hd10_m2d_seed${SEED}_10k \
    model=moirai2_small \
    model.log_on_step=true \
    data=lotsa_v1_moirai2 \
    trainer.max_epochs=100 \
    trainer.precision=bf16-mixed \
    tf32=false \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    train_dataloader.num_workers=8 \
    model.num_warmup_steps=1000 \
    trainer.enable_progress_bar=true \
    seed=$SEED \
    model.anomaly_zscore_threshold=8.0 \
    model.anomaly_variance_ratio_threshold=0.0 \
    model.module_kwargs.time_precondition_enabled=true \
    model.module_kwargs.time_precondition_type=chebyshev \
    model.module_kwargs.time_precondition_degree=4 \
    model.module_kwargs.time_precondition_stride=16 \
    model.module_kwargs.time_precondition_hint_mode=true \
    model.module_kwargs.hint_dropout=0.1

  echo ""
  echo "--- Training baseline seed=$SEED (moirai2 data) ---"
  echo "Time: $(date)"
  python -m cli.train -cp conf/pretrain \
    run_name=baseline_m2d_seed${SEED}_10k \
    model=moirai2_small \
    model.log_on_step=true \
    data=lotsa_v1_moirai2 \
    trainer.max_epochs=100 \
    trainer.precision=bf16-mixed \
    tf32=false \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    train_dataloader.num_workers=8 \
    model.num_warmup_steps=1000 \
    trainer.enable_progress_bar=true \
    seed=$SEED \
    model.anomaly_zscore_threshold=8.0 \
    model.anomaly_variance_ratio_threshold=0.0
done

echo ""
echo "=== Training complete, starting eval ==="

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for SEED in 0 1 2; do
  for MODEL in hd10_m2d_seed${SEED}_10k baseline_m2d_seed${SEED}_10k; do
    ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
    echo ""
    echo "--- Evaluating ${MODEL} ---"
    if [ ! -f "$ckpt" ]; then
      echo "SKIP: checkpoint not found at $ckpt"
      continue
    fi
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
      --checkpoint "$ckpt" --context-length 4000 --batch-size 64
  done
done

echo "=== All runs complete: $(date) ==="
