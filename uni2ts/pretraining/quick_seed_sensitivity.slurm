#!/bin/bash
#SBATCH --job-name=seed_ms46
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=spectralssmtorch
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/seed_ms46_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/seed_ms46_%j.err

# SEED SENSITIVITY: Train ms46 + baseline with 3 different seeds
# Shows results are robust to initialization, not seed-dependent
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

echo "=== Seed Sensitivity Analysis ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for SEED in 0 1 2; do
  echo ""
  echo "--- Training ms46 seed=$SEED ---"
  echo "Time: $(date)"
  python -m cli.train -cp conf/pretrain \
    run_name=ms46_seed${SEED}_10k \
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
    seed=$SEED \
    model.anomaly_zscore_threshold=8.0 \
    model.module_kwargs.time_precondition_enabled=true \
    model.module_kwargs.time_precondition_type=chebyshev \
    model.module_kwargs.time_precondition_degree=4 \
    model.module_kwargs.time_precondition_stride=16 \
    model.module_kwargs.time_precondition_hint_mode=true \
    +model.module_kwargs.time_precondition_extra_hints="6:16"

  echo ""
  echo "--- Training baseline seed=$SEED ---"
  echo "Time: $(date)"
  python -m cli.train -cp conf/pretrain \
    run_name=baseline_seed${SEED}_10k \
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
    seed=$SEED \
    model.anomaly_zscore_threshold=8.0
done

echo "All training complete: $(date)"

# Auto-eval all 6 models
cd /scratch/gpfs/EHAZAN/jh1161/gifteval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUTPUTS=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted

for SEED in 0 1 2; do
  for run in ms46_seed${SEED}_10k baseline_seed${SEED}_10k; do
    CKPT="${OUTPUTS}/${run}/checkpoints/epoch_99-step_10000.ckpt"
    if [ -f "$CKPT" ]; then
      echo "--- Evaluating: $run ---"
      python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: $run"
    else
      echo "SKIP: No checkpoint for $run"
    fi
  done
done
echo "End: $(date)"
