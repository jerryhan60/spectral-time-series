#!/bin/bash
#SBATCH --job-name=rand_h46
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/rand_h46_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/rand_h46_%j.err

# ABLATION: Random hint control — same ms46 architecture (in_proj=64) but hints = random Gaussian noise
# Tests whether the specific polynomial structure matters (vs arbitrary extra signal)
# Expected: if random hint ≈ baseline, the Chebyshev structure is what drives improvement
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Random hint ablation (ms46 architecture)"
python -m cli.train -cp conf/pretrain \
  run_name=random_hint_ms46_10k \
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
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_ablation=random \
  +model.module_kwargs.time_precondition_extra_hints="6:16"
echo "Training done: $(date)"

# Auto-eval
cd /scratch/gpfs/EHAZAN/jh1161/gifteval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/random_hint_ms46_10k/checkpoints/epoch_99-step_10000.ckpt"
if [ -f "$CKPT" ]; then
  echo "--- Evaluating random hint @ 10K ---"
  python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
fi
echo "End: $(date)"
