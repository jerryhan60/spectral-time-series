#!/bin/bash
#SBATCH --job-name=s13_res
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=hazan_intern --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/s13_resume_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/s13_resume_%j.err

# Resume seed=13 runs that timed out (ms46 got to ~epoch 40, bl+dup not started)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

echo "=== Seed 13 Resume ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# Resume ms46 seed=13 from last checkpoint
LAST_CKPT=$(ls -t outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms46_seed13_10k/checkpoints/*.ckpt 2>/dev/null | head -1)
echo "--- Resuming ms46 seed=13 from $LAST_CKPT ---"
python -m cli.train -cp conf/pretrain \
  run_name=ms46_seed13_10k \
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
  seed=13 \
  model.anomaly_zscore_threshold=8.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  +model.module_kwargs.time_precondition_extra_hints="6:16" \
  +ckpt_path="$LAST_CKPT"

echo ""
echo "--- Training baseline seed=13 ---"
python -m cli.train -cp conf/pretrain \
  run_name=baseline_seed13_10k \
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
  seed=13 \
  model.anomaly_zscore_threshold=8.0

echo "Seed 13 resume complete: $(date)"
