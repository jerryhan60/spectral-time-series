#!/bin/bash
#
# Parallel Pretraining Sweep for Chebyshev Preconditioning
# Launches 10 independent SLURM jobs for degrees 1-10
#
# Usage:
#   bash pretraining/pretrain_moirai_precond_sweep.sh
#
# This will submit 10 jobs in parallel, one for each polynomial degree.
# Jobs will run as GPU resources become available.

echo "=========================================="
echo "Launching Chebyshev Preconditioning Sweep"
echo "Degrees: 1 through 10"
echo "=========================================="
echo ""

# Ensure logs directory exists
mkdir -p logs

# Store job IDs for tracking
declare -a JOB_IDS

# Submit jobs for degrees 1-10
for DEGREE in {1..10}; do
    echo "Submitting job for Chebyshev degree $DEGREE..."

    # Submit the job and capture the job ID
    # Override output/error filenames to include degree
    JOB_OUTPUT=$(sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=$DEGREE \
                        --job-name=precond_cheb_d${DEGREE} \
                        --output=logs/pretrain_precond_cheb_d${DEGREE}_%j.out \
                        --error=logs/pretrain_precond_cheb_d${DEGREE}_%j.err \
                        pretraining/pretrain_moirai_precond.slurm)

    # Extract job ID from sbatch output
    JOB_ID=$(echo $JOB_OUTPUT | awk '{print $4}')
    JOB_IDS+=($JOB_ID)

    echo "  → Job ID: $JOB_ID"
    echo ""
done

echo "=========================================="
echo "All jobs submitted successfully!"
echo "=========================================="
echo ""
echo "Job IDs:"
for i in "${!JOB_IDS[@]}"; do
    DEGREE=$((i + 1))
    echo "  Degree $DEGREE: ${JOB_IDS[$i]}"
done

echo ""
echo "To monitor all jobs:"
echo "  squeue -u \$USER"
echo ""
echo "To check specific job outputs:"
echo "  tail -f logs/pretrain_precond_cheb_d5_*.out  # for degree 5"
echo "  ls -lth logs/pretrain_precond_cheb_d*_*.out  # list all"
echo ""
echo "To cancel all jobs:"
echo "  scancel ${JOB_IDS[@]}"
echo ""
echo "=========================================="
