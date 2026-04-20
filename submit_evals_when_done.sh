#!/bin/bash
# Auto-submit GIFT-Eval for training jobs when they complete
# Usage: bash submit_evals_when_done.sh

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
EVAL_SCRIPT="/scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.slurm"

declare -A JOBS
JOBS[5008831]="q_c08_d10"
JOBS[5008832]="q_d6_d10"
JOBS[5008571]="q_c08_d10_ailab"
JOBS[5008572]="q_d6_d10_ailab"
JOBS[5008647]="q_d8"
JOBS[5008648]="q_d6_learn"
JOBS[5008649]="q_d4_drop05"
JOBS[5008656]="q_d6_sep"

for JID in "${!JOBS[@]}"; do
  NAME="${JOBS[$JID]}"
  STATE=$(squeue -j $JID --noheader --format="%T" 2>/dev/null)
  if [ -z "$STATE" ]; then
    # Job completed - find checkpoint
    CKPT=$(find outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ -path "*${NAME}*" -name "epoch_99-step_10000.ckpt" 2>/dev/null | sort -r | head -1)
    if [ -n "$CKPT" ]; then
      echo "Job $JID ($NAME) COMPLETED. Checkpoint: $CKPT"
      CHECKPOINT="$PWD/$CKPT" sbatch --partition=ailab --account=ehazan "$EVAL_SCRIPT"
      echo "  Submitted GIFT-Eval"
    else
      echo "Job $JID ($NAME) COMPLETED but no checkpoint found"
    fi
  else
    echo "Job $JID ($NAME): $STATE"
  fi
done
