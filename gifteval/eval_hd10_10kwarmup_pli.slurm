#!/bin/bash
#SBATCH --job-name=ge_hd10_10kw
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=2:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_hd10_10kw_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_hd10_10kw_%j.err

# GIFT-Eval evaluation for hint d=4 + 10% dropout with 10K warmup (baseline-matched LR)
# Resubmitted to pli partition
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a
export HF_HUB_OFFLINE=1

echo "=== GIFT-Eval Full Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"

# Find the checkpoint from the training run (run name starts with m2_hd10_100k_10kwarmup_)
CKPT_DIR=$(ls -dt uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_hd10_100k_10kwarmup_*/checkpoints 2>/dev/null | head -1)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint directory found for m2_hd10_100k_10kwarmup_*"
    exit 1
fi

CHECKPOINT="${CKPT_DIR}/epoch_999-step_100000.ckpt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Final checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls "$CKPT_DIR"/*.ckpt 2>/dev/null
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo ""

python gifteval/eval_gifteval.py \
  --checkpoint "$CHECKPOINT" \
  --context-length 4000 \
  --patch-size 32 \
  --batch-size 64

echo "=== Evaluation Complete ==="
echo "Results in: /scratch/gpfs/EHAZAN/jh1161/gifteval/results/"
