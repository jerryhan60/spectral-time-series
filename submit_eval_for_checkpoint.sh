#!/bin/bash
# Usage: ./submit_eval_for_checkpoint.sh <checkpoint_path> [partition]
# Submits GIFT-Eval job for a given checkpoint on both ailab and pli

CKPT="$1"
PARTITION="${2:-both}"  # ailab, pli, or both

if [ -z "$CKPT" ]; then
  echo "Usage: $0 <checkpoint_path> [ailab|pli|both]"
  exit 1
fi

if [ ! -f "$CKPT" ]; then
  echo "Error: checkpoint not found: $CKPT"
  exit 1
fi

echo "Submitting GIFT-Eval for: $CKPT"

if [ "$PARTITION" = "ailab" ] || [ "$PARTITION" = "both" ]; then
  CHECKPOINT="$CKPT" sbatch --partition=ailab --account=ehazan /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.slurm
fi

if [ "$PARTITION" = "pli" ] || [ "$PARTITION" = "both" ]; then
  CHECKPOINT="$CKPT" sbatch --partition=pli --account=eladgroup /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.slurm
fi
