#!/bin/bash
#
# Helper script to evaluate all checkpoints from the sweep
#
# Usage:
#   bash pretraining/evaluate_all_checkpoints.sh
#
# This will:
# 1. Find all checkpoints for degrees 1-10
# 2. Submit evaluation jobs for each checkpoint
# 3. Output job IDs for tracking

echo "=========================================="
echo "Evaluating All Checkpoints"
echo "=========================================="
echo ""

# Check if evaluation script exists
EVAL_SCRIPT="eval/eval_precond_comprehensive.slurm"
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

# Store job IDs
declare -a JOB_IDS

SUBMITTED_COUNT=0
MISSING_COUNT=0

for d in {1..10}; do
    # Find the most recent checkpoint for this degree
    CKPT=$(find uni2ts/outputs -path "*precond_chebyshev_d${d}_*/checkpoints/last.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -n "$CKPT" ]; then
        echo "Submitting evaluation for degree $d..."
        echo "  Checkpoint: $CKPT"

        # Submit evaluation job
        JOB_OUTPUT=$(sbatch --export=MODEL_PATH=$CKPT,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=$d \
                            --job-name=eval_d${d} \
                            $EVAL_SCRIPT)

        # Extract job ID
        JOB_ID=$(echo $JOB_OUTPUT | awk '{print $4}')
        JOB_IDS+=($JOB_ID)

        echo "  Job ID: $JOB_ID"
        echo ""

        SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
    else
        echo "Skipping degree $d: No checkpoint found"
        echo ""
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

echo "=========================================="
echo "Evaluation Jobs Submitted"
echo "=========================================="
echo "  Submitted: $SUBMITTED_COUNT jobs"
echo "  Skipped: $MISSING_COUNT (no checkpoints)"
echo ""

if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo "Job IDs:"
    for i in "${!JOB_IDS[@]}"; do
        DEGREE=$((i + 1))
        echo "  Degree $DEGREE: ${JOB_IDS[$i]}"
    done

    echo ""
    echo "To monitor jobs:"
    echo "  squeue -u \$USER"
    echo ""
    echo "To cancel all evaluation jobs:"
    echo "  scancel ${JOB_IDS[@]}"
fi

echo ""
echo "=========================================="
