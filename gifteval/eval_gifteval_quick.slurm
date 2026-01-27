#!/bin/bash
#SBATCH --job-name=gifteval_quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/gifteval_quick_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/gifteval_quick_%j.err

# GIFT-Eval Quick Evaluation (8 representative datasets)
# Usage:
#   sbatch eval_gifteval_quick.slurm                                         # Default checkpoint
#   sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt eval_gifteval_quick.slurm  # Custom checkpoint
#   sbatch --export=MODEL=moirai-1.1-R-small eval_gifteval_quick.slurm       # HuggingFace model

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

echo "=== GIFT-Eval Quick Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "Time: $(date)"
echo ""

# Determine checkpoint or model
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
    python eval_gifteval.py --checkpoint "$CHECKPOINT" --quick --batch-size 64
elif [ -n "$MODEL" ]; then
    echo "Model: $MODEL"
    python eval_gifteval.py --model "$MODEL" --quick --batch-size 64
else
    # Default: use latest checkpoint from training
    DEFAULT_CKPT=$(ls -t /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/*/checkpoints/*.ckpt 2>/dev/null | head -n1)
    if [ -n "$DEFAULT_CKPT" ]; then
        echo "Using latest checkpoint: $DEFAULT_CKPT"
        python eval_gifteval.py --checkpoint "$DEFAULT_CKPT" --quick --batch-size 64
    else
        echo "No checkpoint specified and no default found."
        echo "Usage: sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt eval_gifteval_quick.slurm"
        exit 1
    fi
fi

echo ""
echo "=== Evaluation Complete ==="
echo "Results in: /scratch/gpfs/EHAZAN/jh1161/gifteval/results/"
