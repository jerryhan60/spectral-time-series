#!/bin/bash
# Run after training job 5028842 completes
# Usage: bash submit_stu_eval.sh

# Find the latest checkpoint
CKPT_DIR="uni2ts/outputs/pretrain/moirai2_small_stu/lotsa_v1_unweighted"
CKPT=$(find "$CKPT_DIR" -name "*.ckpt" -path "*/stu_hybrid_10k/*" | sort -t/ -k10 | tail -1)

if [ -z "$CKPT" ]; then
    echo "No checkpoint found in $CKPT_DIR/*/stu_hybrid_10k/"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"

# Submit GIFT-Eval
export CHECKPOINT="$CKPT"
sbatch --export=CHECKPOINT /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.slurm
