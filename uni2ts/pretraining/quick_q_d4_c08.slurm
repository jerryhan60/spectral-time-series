#!/bin/bash
#SBATCH --job-name=q_d4_c08
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=logs/q_d4_c08_%j.out --error=logs/q_d4_c08_%j.err

# d=4 with SCALED coefficients (0.8x standard Chebyshev): [0, -0.8, 0, 0.1]
# Combines optimal scaling (-0.8 primary) with d=4 correction term
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: d=4 scaled coeffs [0,-0.8,0,0.1] s=16"
python -m cli.train -cp conf/pretrain run_name=q_d4_c08_$(date +%Y%m%d_%H%M%S) model=moirai2_small data=lotsa_v1_unweighted trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=4 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=42 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true 'model.module_kwargs.time_precondition_coeffs_init=[0.0,-0.8,0.0,0.1]'
echo "End: $(date)"
