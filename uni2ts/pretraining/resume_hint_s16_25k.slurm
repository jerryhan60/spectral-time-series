#!/bin/bash
#SBATCH --job-name=m2_hint25k_r
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=logs/m2_hint25k_resume_%j.out --error=logs/m2_hint25k_resume_%j.err

# Resume 25K hint mode training from last checkpoint
# Original job 4914536 will timeout at ~epoch 212/250

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

# Find the latest checkpoint
CKPT_DIR=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_hint_s16_25k_20260219_190429/checkpoints
LAST_CKPT=$(ls -t $CKPT_DIR/*.ckpt | head -1)
echo "Resuming from: $LAST_CKPT"
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: hint mode s16 d=5 25K RESUME"

python -m cli.train -cp conf/pretrain \
  run_name=m2_hint_s16_25k_20260219_190429 model=moirai2_small data=lotsa_v1_unweighted \
  trainer.max_epochs=250 trainer.precision=bf16-mixed tf32=false \
  train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=42 \
  model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 \
  model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=5 model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  ckpt_path=$LAST_CKPT
echo "End: $(date)"
