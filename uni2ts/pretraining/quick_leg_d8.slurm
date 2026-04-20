#!/bin/bash
#SBATCH --job-name=q_leg_d8
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=logs/q_leg_d8_%j.out --error=logs/q_leg_d8_%j.err

# Legendre d=8 hint s=16
# Legendre d=8 max|c|=1.87 vs Chebyshev d=8 max|c|=2.0
# Where Chebyshev d=8 degraded (1.2216), Legendre might recover
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Legendre d=8 s=16 hint"
python -m cli.train -cp conf/pretrain run_name=q_leg_d8_$(date +%Y%m%d_%H%M%S) model=moirai2_small data=lotsa_v1_unweighted trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=4 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=42 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=legendre model.module_kwargs.time_precondition_degree=8 model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true
echo "End: $(date)"
