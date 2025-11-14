#!/bin/bash
#
# Helper script to submit evaluation jobs for all frequencies (yearly, quarterly, monthly)
# Each frequency is submitted as a separate job for parallel execution
#
# Usage: bash submit_eval_all_frequencies.sh /path/to/checkpoint.ckpt
#

if [ $# -eq 0 ]; then
    echo "ERROR: No checkpoint path provided"
    echo ""
    echo "Usage: bash submit_eval_all_frequencies.sh /path/to/checkpoint.ckpt"
    echo ""
    echo "Example:"
    echo "  bash submit_eval_all_frequencies.sh outputs/precond_cheb_5_20251101_143052/checkpoints/last.ckpt"
    exit 1
fi

CHECKPOINT_PATH=$1

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    exit 1
fi

echo "=========================================="
echo "Submitting Evaluation Jobs"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit yearly evaluation
echo "Submitting YEARLY evaluation..."
YEARLY_JOB=$(sbatch --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,FREQUENCY=yearly \
             --parsable eval_moirai_by_frequency.slurm)
echo "  Job ID: $YEARLY_JOB"

# Submit quarterly evaluation
echo "Submitting QUARTERLY evaluation..."
QUARTERLY_JOB=$(sbatch --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,FREQUENCY=quarterly \
                --parsable eval_moirai_by_frequency.slurm)
echo "  Job ID: $QUARTERLY_JOB"

# Submit monthly evaluation
echo "Submitting MONTHLY evaluation..."
MONTHLY_JOB=$(sbatch --export=CHECKPOINT_PATH=$CHECKPOINT_PATH,FREQUENCY=monthly \
              --parsable eval_moirai_by_frequency.slurm)
echo "  Job ID: $MONTHLY_JOB"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Job IDs:"
echo "  Yearly:    $YEARLY_JOB"
echo "  Quarterly: $QUARTERLY_JOB"
echo "  Monthly:   $MONTHLY_JOB"
echo ""
echo "Check status: squeue -u $USER"
echo "View logs:    tail -f logs/eval_by_freq_*.out"
echo "=========================================="
