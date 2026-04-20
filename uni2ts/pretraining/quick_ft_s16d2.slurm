#!/bin/bash
#SBATCH --job-name=q_ft_d2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:30:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=logs/q_ft_d2_%j.out --error=logs/q_ft_d2_%j.err
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
BASELINE_CKPT="outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_baseline_25k_20260217_031428/checkpoints/epoch_249-step_25000.ckpt"
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Fine-tune baseline with stride=16 d=2"
python -m cli.train -cp conf/pretrain \
  run_name=q_ft_s16d2_$(date +%Y%m%d_%H%M%S) model=moirai2_small data=lotsa_v1_unweighted \
  trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false \
  train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=42 \
  model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 \
  model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=2 model.module_kwargs.time_precondition_stride=16 \
  baseline_ckpt_path="$BASELINE_CKPT"
echo "End: $(date)"
