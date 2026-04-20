#!/bin/bash
#SBATCH --job-name=gifteval_m2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/gifteval_moirai2_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/gifteval_moirai2_%j.err

# GIFT-Eval Evaluation for Moirai2 checkpoints
# Usage:
#   sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt eval_moirai2.slurm
#   sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt,FULL=1 eval_moirai2.slurm

set -e

# Load modules
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate environment
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

# Set offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /scratch/gpfs/EHAZAN/jh1161/gifteval

echo "=== GIFT-Eval Moirai2 Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "Time: $(date)"
echo ""

if [ -z "$CHECKPOINT" ]; then
    # Find latest Moirai2 checkpoint
    CHECKPOINT=$(ls -t /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/moirai2_*/checkpoints/*.ckpt 2>/dev/null | head -n1)
    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint specified and no Moirai2 checkpoint found."
        echo "Usage: sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt eval_moirai2.slurm"
        exit 1
    fi
fi

echo "Checkpoint: $CHECKPOINT"

EXTRA_ARGS="--quick"
if [ -n "$FULL" ]; then
    EXTRA_ARGS=""
fi

python eval_gifteval.py \
    --checkpoint "$CHECKPOINT" \
    $EXTRA_ARGS \
    --batch-size 64 \
    --context-length 1000 \
    --patch-size 16

echo ""
echo "=== Evaluation Complete ==="
echo "Results in: /scratch/gpfs/EHAZAN/jh1161/gifteval/results/"
