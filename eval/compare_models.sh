#!/bin/bash
#
# Helper script to run comprehensive comparison between baseline and preconditioned models
#
# This submits all three evaluation jobs:
# 1. Baseline model in original space
# 2. Preconditioned model in original space (reversed)
# 3. Baseline model in preconditioned space (for fair comparison)
# 4. Preconditioned model in preconditioned space (native)
#
# Usage:
#   bash compare_models.sh <baseline_ckpt> <precond_ckpt> <precond_type> <precond_degree>
#
# Example:
#   bash compare_models.sh \
#     /path/to/baseline.ckpt \
#     /path/to/precond_chebyshev_d5.ckpt \
#     chebyshev \
#     5

if [ $# -lt 4 ]; then
    echo "ERROR: Insufficient arguments"
    echo ""
    echo "Usage: bash compare_models.sh <baseline_ckpt> <precond_ckpt> <precond_type> <precond_degree>"
    echo ""
    echo "Arguments:"
    echo "  baseline_ckpt:   Path to baseline model checkpoint"
    echo "  precond_ckpt:    Path to preconditioned model checkpoint"
    echo "  precond_type:    Polynomial type (chebyshev or legendre)"
    echo "  precond_degree:  Polynomial degree (e.g., 5)"
    echo ""
    echo "Example:"
    echo "  bash compare_models.sh \\"
    echo "    /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/.../baseline.ckpt \\"
    echo "    /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/.../precond.ckpt \\"
    echo "    chebyshev \\"
    echo "    5"
    exit 1
fi

BASELINE_CKPT=$1
PRECOND_CKPT=$2
PRECOND_TYPE=$3
PRECOND_DEGREE=$4

# Validate files exist
if [ ! -f "$BASELINE_CKPT" ]; then
    echo "ERROR: Baseline checkpoint not found: $BASELINE_CKPT"
    exit 1
fi

if [ ! -f "$PRECOND_CKPT" ]; then
    echo "ERROR: Preconditioned checkpoint not found: $PRECOND_CKPT"
    exit 1
fi

echo "=========================================="
echo "Comprehensive Model Comparison"
echo "=========================================="
echo "Baseline checkpoint:      $BASELINE_CKPT"
echo "Preconditioned checkpoint: $PRECOND_CKPT"
echo "Preconditioning type:      $PRECOND_TYPE"
echo "Preconditioning degree:    $PRECOND_DEGREE"
echo "=========================================="
echo ""

cd /scratch/gpfs/EHAZAN/jh1161

# Create logs directory if needed
mkdir -p logs

echo "Submitting evaluation jobs..."
echo ""

# Job 1: Baseline in original space
echo "1. Baseline model → Original space"
JOB1=$(sbatch --export=MODEL_PATH=$BASELINE_CKPT eval/eval_comprehensive.slurm 2>&1 | grep -oE "[0-9]+")
echo "   Submitted job: $JOB1"
echo ""

# Job 2: Preconditioned in original space (with reversal)
echo "2. Preconditioned model → Original space (reversed)"
JOB2=$(sbatch --export=MODEL_PATH=$PRECOND_CKPT eval/eval_comprehensive.slurm 2>&1 | grep -oE "[0-9]+")
echo "   Submitted job: $JOB2"
echo ""

# Job 3: Baseline in preconditioned space (post-hoc transformation)
echo "3. Baseline model → Preconditioned space (post-hoc)"
JOB3=$(sbatch --export=MODEL_PATH=$BASELINE_CKPT,PRECOND_TYPE=$PRECOND_TYPE,PRECOND_DEGREE=$PRECOND_DEGREE \
  eval/eval_baseline_in_precond_space.slurm 2>&1 | grep -oE "[0-9]+")
echo "   Submitted job: $JOB3"
echo ""

# Job 4: Preconditioned in preconditioned space (native)
echo "4. Preconditioned model → Preconditioned space (native)"
JOB4=$(sbatch --export=MODEL_PATH=$PRECOND_CKPT,PRECOND_TYPE=$PRECOND_TYPE,PRECOND_DEGREE=$PRECOND_DEGREE \
  eval/eval_precond_comprehensive.slurm 2>&1 | grep -oE "[0-9]+")
echo "   Submitted job: $JOB4"
echo ""

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/eval_comprehensive_${JOB1}.out"
echo "  tail -f logs/eval_comprehensive_${JOB2}.out"
echo "  tail -f logs/eval_baseline_precond_space_${JOB3}.out"
echo "  tail -f logs/eval_precond_comprehensive_${JOB4}.out"
echo ""
echo "Results will be in:"
echo "  eval/outputs/"
echo ""
echo "When complete, compare CSVs:"
echo "  1. Original space: Job $JOB1 vs Job $JOB2"
echo "  2. Preconditioned space: Job $JOB3 vs Job $JOB4"
echo "=========================================="
