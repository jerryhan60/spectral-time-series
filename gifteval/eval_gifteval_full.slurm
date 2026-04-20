#!/bin/bash
#SBATCH --job-name=gifteval_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/gifteval_full_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/gifteval_full_%j.err

# GIFT-Eval Full Evaluation (97 dataset configurations)
# Usage:
#   sbatch eval_gifteval_full.slurm                                         # Default checkpoint
#   sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt eval_gifteval_full.slurm  # Custom checkpoint
#   sbatch --export=MODEL=moirai-1.1-R-small eval_gifteval_full.slurm       # HuggingFace model

set -e

# Load modules
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate environment
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

# Set offline mode (no internet on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Change to working directory
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

echo "=== GIFT-Eval Full Evaluation (97 configs) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "Time: $(date)"
echo ""

# Determine checkpoint or model
if [ -n "$CHECKPOINT" ]; then
    # Extract model name from path for display
    MODEL_DIR=$(dirname $(dirname $(dirname "$CHECKPOINT")))
    MODEL_NAME=$(basename "$MODEL_DIR")
    CKPT_NAME=$(basename "$CHECKPOINT" .ckpt)

    echo "=========================================="
    echo "Model: $MODEL_NAME"
    echo "Checkpoint: $CKPT_NAME"
    echo "Full path: $CHECKPOINT"
    echo "=========================================="
    echo ""

    python eval_gifteval.py --checkpoint "$CHECKPOINT" --batch-size 64
elif [ -n "$MODEL" ]; then
    echo "=========================================="
    echo "Model: $MODEL (HuggingFace)"
    echo "=========================================="
    echo ""

    python eval_gifteval.py --model "$MODEL" --batch-size 64
else
    # Default: use latest checkpoint from training
    DEFAULT_CKPT=$(ls -t /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/*/checkpoints/*.ckpt 2>/dev/null | head -n1)
    if [ -n "$DEFAULT_CKPT" ]; then
        echo "Using latest checkpoint: $DEFAULT_CKPT"
        python eval_gifteval.py --checkpoint "$DEFAULT_CKPT" --batch-size 64
    else
        echo "No checkpoint specified and no default found."
        echo "Usage: sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt eval_gifteval_full.slurm"
        exit 1
    fi
fi

echo ""
echo "=== Evaluation Complete ==="
echo "Results in: /scratch/gpfs/EHAZAN/jh1161/gifteval/results/"
echo ""
echo "Markdown reports saved as: report_*.md"
