#!/bin/bash
# Submit FEV bench and GIFT-Eval evaluations for stride=16 experiments
# Run this AFTER training jobs 4829784, 4829785, 4829786 complete

set -euo pipefail

# Find the checkpoints
OUTPUTS="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted"

echo "Looking for stride=16 and strongL2 checkpoints..."

# Find stride=16 d=4 checkpoint
S16D4_DIR=$(ls -td ${OUTPUTS}/m2_stride16_d4_free_* 2>/dev/null | head -1)
if [[ -z "$S16D4_DIR" ]]; then
    echo "ERROR: No stride16_d4 checkpoint found"
    exit 1
fi
S16D4_CKPT="${S16D4_DIR}/checkpoints/epoch_249-step_25000.ckpt"
echo "stride16_d4: $S16D4_CKPT"

# Find stride=16 d=1 checkpoint
S16D1_DIR=$(ls -td ${OUTPUTS}/m2_stride16_d1_free_* 2>/dev/null | head -1)
if [[ -z "$S16D1_DIR" ]]; then
    echo "ERROR: No stride16_d1 checkpoint found"
    exit 1
fi
S16D1_CKPT="${S16D1_DIR}/checkpoints/epoch_249-step_25000.ckpt"
echo "stride16_d1: $S16D1_CKPT"

# Find strongL2 checkpoint
SL2_DIR=$(ls -td ${OUTPUTS}/m2_s1_strongL2_* 2>/dev/null | head -1)
if [[ -z "$SL2_DIR" ]]; then
    echo "ERROR: No strongL2 checkpoint found"
    exit 1
fi
SL2_CKPT="${SL2_DIR}/checkpoints/epoch_249-step_25000.ckpt"
echo "strongL2: $SL2_CKPT"

# Verify all checkpoints exist
for ckpt in "$S16D4_CKPT" "$S16D1_CKPT" "$SL2_CKPT"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "WAITING: Checkpoint not yet created: $ckpt"
        echo "Training may still be running. Check with: squeue -u \$USER"
        exit 1
    fi
done

echo ""
echo "All checkpoints found. Submitting evaluations..."

# Submit FEV bench evals
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

export CKPT_PATH MODEL_NAME

CKPT_PATH="$S16D4_CKPT" MODEL_NAME="stride16_d4_25k" sbatch eval_fev_bench.slurm
echo "Submitted FEV bench: stride16_d4"

CKPT_PATH="$S16D1_CKPT" MODEL_NAME="stride16_d1_25k" sbatch eval_fev_bench.slurm
echo "Submitted FEV bench: stride16_d1"

CKPT_PATH="$SL2_CKPT" MODEL_NAME="strongL2_25k" sbatch eval_fev_bench.slurm
echo "Submitted FEV bench: strongL2"

# Submit GIFT-Eval
cd /scratch/gpfs/EHAZAN/jh1161

CHECKPOINT="$S16D4_CKPT" sbatch gifteval/eval_gifteval.slurm
echo "Submitted GIFT-Eval: stride16_d4"

CHECKPOINT="$S16D1_CKPT" sbatch gifteval/eval_gifteval.slurm
echo "Submitted GIFT-Eval: stride16_d1"

CHECKPOINT="$SL2_CKPT" sbatch gifteval/eval_gifteval.slurm
echo "Submitted GIFT-Eval: strongL2"

echo ""
echo "All evaluations submitted!"
