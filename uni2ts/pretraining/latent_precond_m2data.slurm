#!/bin/bash
#SBATCH --job-name=lat_prec
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=spectralssmtorch --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/lat_prec_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/lat_prec_%j.err

# Latent preconditioning experiments on lotsa_v1_moirai2 (seed=0 screen)
# Tests true USP: precondition hidden reprs BEFORE transformer, reverse AFTER
# This is the Marsden-Hazan approach applied in representation space
# Also tests: latent + hint combined, and latent-only at different degrees
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"

echo "=== Latent Preconditioning Experiments (lotsa_v1_moirai2, seed=0) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Latent-only: Chebyshev d=4, stride=1 (apply to every token)
echo -e "\n--- lat_c4s1_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=lat_c4s1_m2d_seed0_10k $COMMON \
  model.module_kwargs.latent_precondition_enabled=true \
  model.module_kwargs.latent_precondition_type=chebyshev \
  model.module_kwargs.latent_precondition_degree=4 \
  model.module_kwargs.latent_precondition_stride=1

# 2. Latent-only: Chebyshev d=4, stride=4
echo -e "\n--- lat_c4s4_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=lat_c4s4_m2d_seed0_10k $COMMON \
  model.module_kwargs.latent_precondition_enabled=true \
  model.module_kwargs.latent_precondition_type=chebyshev \
  model.module_kwargs.latent_precondition_degree=4 \
  model.module_kwargs.latent_precondition_stride=4

# 3. Latent-only: Chebyshev d=6, stride=1
echo -e "\n--- lat_c6s1_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=lat_c6s1_m2d_seed0_10k $COMMON \
  model.module_kwargs.latent_precondition_enabled=true \
  model.module_kwargs.latent_precondition_type=chebyshev \
  model.module_kwargs.latent_precondition_degree=6 \
  model.module_kwargs.latent_precondition_stride=1

# 4. Combined: Hint d=4 (hd10) + Latent d=4 (true USP for hidden states)
echo -e "\n--- hd10_lat4_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_lat4_m2d_seed0_10k $COMMON \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1 \
  model.module_kwargs.latent_precondition_enabled=true \
  model.module_kwargs.latent_precondition_type=chebyshev \
  model.module_kwargs.latent_precondition_degree=4 \
  model.module_kwargs.latent_precondition_stride=1

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for MODEL in lat_c4s1_m2d_seed0_10k lat_c4s4_m2d_seed0_10k lat_c6s1_m2d_seed0_10k hd10_lat4_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== All latent precond runs complete: $(date) ==="
