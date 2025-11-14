#!/bin/bash
#
# Helper script to submit a parameter sweep of preconditioning experiments
# This script submits multiple jobs with different preconditioning configurations
#
# Usage: bash submit_precond_sweep.sh
#

echo "=========================================="
echo "Submitting Preconditioning Parameter Sweep"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Baseline (no preconditioning) for comparison
echo "Submitting baseline (no preconditioning)..."
sbatch pretrain_moirai.slurm
echo ""

# Default preconditioning (Chebyshev degree 5)
echo "Submitting default preconditioning (Chebyshev degree 5)..."
sbatch pretrain_moirai_precond_default.slurm
echo ""

# Degree sweep with Chebyshev
echo "Submitting Chebyshev degree sweep..."
for degree in 2 3 5 7 10; do
    echo "  - Chebyshev degree $degree"
    sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=$degree,RUN_SUFFIX=_sweep \
           pretrain_moirai_precond.slurm
done
echo ""

# Legendre polynomial comparison at degree 5
echo "Submitting Legendre comparison (degree 5)..."
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=5,RUN_SUFFIX=_sweep \
       pretrain_moirai_precond.slurm
echo ""

echo "=========================================="
echo "All jobs submitted!"
echo "Check job status with: squeue -u $USER"
echo "View logs in: logs/"
echo "=========================================="
