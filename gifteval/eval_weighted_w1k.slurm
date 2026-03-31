#!/bin/bash
#SBATCH --job-name=ge_w1k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_w1k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_w1k_%j.err

# Evaluate weighted LOTSA w1k baseline + hint (10K steps, warmup=1K)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== Weighted LOTSA w1k Evaluations ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Weighted baseline w1k
CKPT=$(ls -t uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_weighted/wt_base_w1k_*/checkpoints/epoch_99-step_10000.ckpt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    echo ""
    echo "=== Weighted Baseline w1k ==="
    echo "Checkpoint: $CKPT"
    python gifteval/eval_gifteval.py --checkpoint "$CKPT"
    echo "Done: $(date)"
else
    echo "ERROR: Weighted baseline w1k checkpoint not found"
fi

# 2. Weighted hint w1k
CKPT=$(ls -t uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_weighted/wt_hint_w1k_*/checkpoints/epoch_99-step_10000.ckpt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    echo ""
    echo "=== Weighted Hint w1k (ms d=4+d=6) ==="
    echo "Checkpoint: $CKPT"
    python gifteval/eval_gifteval.py --checkpoint "$CKPT"
    echo "Done: $(date)"
else
    echo "ERROR: Weighted hint w1k checkpoint not found"
fi

echo ""
echo "=== All w1k evals complete ==="
echo "End: $(date)"
