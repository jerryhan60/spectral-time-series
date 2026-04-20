#!/bin/bash
#SBATCH --job-name=hd_lr2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/hd_lr2_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/hd_lr2_%j.err

# HD10 with higher LR (2e-3 and 3e-3) — fair comparison with BL@LR=2e-3
# Run: sbatch --partition=pli --account=eladgroup --qos=pli-low hd10_lr2e3_10k.slurm

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

SEED=0
COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 seed=${SEED}"
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.time_precondition_degree=4 model.module_kwargs.hint_dropout=0.1"

echo "=== HD10 Higher LR ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# HD10 @ LR=2e-3
echo -e "\n--- hd10_lr2e3_s0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_lr2e3_s0_10k $COMMON $HINT model.lr=2e-3

# HD10 @ LR=3e-3
echo -e "\n--- hd10_lr3e3_s0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_lr3e3_s0_10k $COMMON $HINT model.lr=3e-3

echo "=== Training complete, starting eval ==="

cd /scratch/gpfs/EHAZAN/jh1161
BASE="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2"
for MODEL in hd10_lr2e3_s0_10k hd10_lr3e3_s0_10k; do
  CKPT="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- eval ${MODEL} ---\nTime: $(date)"
  python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
done

echo "=== HD10 LR complete: $(date) ==="
