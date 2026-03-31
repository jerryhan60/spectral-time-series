#!/bin/bash
#SBATCH --job-name=seed_extra
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=18:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/seed_extra_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/seed_extra_%j.err

# Train 2 more seeds (7, 13) for ms46, baseline, and duplicate ablation
# Existing seeds: 0, 1, 44 (ms46+baseline), 42 (original runs + duplicate)
# Total after this: 5 seeds each for ms46, baseline; 2+ for duplicate
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

echo "=== Extra Seed Training ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for SEED in 7 13; do
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

  echo ""
  echo "--- Training duplicate ablation seed=$SEED ---"
  echo "Time: $(date)"
  python -m cli.train -cp conf/pretrain \
    run_name=ablation_duplicate_seed${SEED}_10k \
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
    model.module_kwargs.hint_ablation=duplicate
done

echo "All extra seed training complete: $(date)"
