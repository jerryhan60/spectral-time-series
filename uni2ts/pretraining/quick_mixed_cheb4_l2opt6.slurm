#!/bin/bash
#SBATCH --job-name=q_mix46
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=logs/q_mix46_%j.out --error=logs/q_mix46_%j.err

# Mixed family: Chebyshev d=4 primary + L2-optimized d=6 extra
# Best of each: Cheb d=4 is strong primary (1.1944), L2-opt d=6 beats Cheb d=6 (1.1784 vs 1.1836)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: mixed Cheb d=4 + L2-opt d=6"
python -m cli.train -cp conf/pretrain run_name=q_mix46_$(date +%Y%m%d_%H%M%S) model=moirai2_small data=lotsa_v1_unweighted trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=4 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=42 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.hint_dropout=0.0 '+model.module_kwargs.time_precondition_extra_hints=l2_optimized:6:16'
echo "End: $(date)"
