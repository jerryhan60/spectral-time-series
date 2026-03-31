#!/bin/bash
#SBATCH --job-name=ms46d_10k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=08:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ms46d_10k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ms46d_10k_%j.err

# ms d=4+d=6 hint + general model dropout (10%) + hint dropout (10%)
# Tests if adding model-level dropout regularization on top of hint dropout
# further improves generalization, especially for non-periodic datasets
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: ms d=4+d=6 + model dropout 0.1, 10K"
python -m cli.train -cp conf/pretrain \
  run_name=ms46_modeldrop_10k_$(date +%Y%m%d_%H%M%S) \
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
  model.module_kwargs.dropout_p=0.1 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1 \
  +model.module_kwargs.time_precondition_extra_hints="6:16"
echo "End: $(date)"
