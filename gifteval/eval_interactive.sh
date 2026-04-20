#!/bin/bash
# Interactive GIFT-Eval evaluation script
# Run from a GPU node (salloc session)
#
# Usage:
#   ./eval_interactive.sh /path/to/checkpoint.ckpt          # Quick eval
#   ./eval_interactive.sh /path/to/checkpoint.ckpt --full   # Full eval
#   ./eval_interactive.sh moirai-1.1-R-small --hf           # HuggingFace model

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

# Parse arguments
if [ "$2" == "--full" ]; then
    echo "Running full GIFT-Eval benchmark..."
    python eval_gifteval.py --checkpoint "$1" --batch-size 32
elif [ "$2" == "--hf" ]; then
    echo "Evaluating HuggingFace model: $1"
    python eval_gifteval.py --model "$1" --quick --batch-size 64
else
    echo "Running quick GIFT-Eval evaluation..."
    python eval_gifteval.py --checkpoint "$1" --quick --batch-size 64
fi

echo ""
echo "Results saved to: /scratch/gpfs/EHAZAN/jh1161/gifteval/results/"
