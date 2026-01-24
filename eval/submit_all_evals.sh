#!/bin/bash
# ============================================================================
# Submit All Evaluation Jobs
# ============================================================================
#
# This script submits all 6 evaluation jobs:
#   - Standard baseline on Monash
#   - Standard baseline on LOTSA in-distribution
#   - Embedding D3 on Monash
#   - Embedding D3 on LOTSA in-distribution
#   - Embedding D4 on Monash
#   - Embedding D4 on LOTSA in-distribution
#
# Usage:
#   ./eval/submit_all_evals.sh           # Submit all jobs
#   ./eval/submit_all_evals.sh monash    # Submit only Monash jobs
#   ./eval/submit_all_evals.sh lotsa     # Submit only LOTSA in-distribution jobs
#
# ============================================================================

cd /scratch/gpfs/EHAZAN/jh1161

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Submitting Evaluation Jobs"
echo "=========================================="
echo ""

FILTER=${1:-all}

if [ "$FILTER" == "all" ] || [ "$FILTER" == "monash" ]; then
    echo "Submitting Monash benchmark evaluations..."
    
    JOB1=$(sbatch --parsable eval/eval_standard_monash.slurm)
    echo "  Standard Monash: Job $JOB1"
    
    JOB2=$(sbatch --parsable eval/eval_embedding_d3_monash.slurm)
    echo "  Embedding D3 Monash: Job $JOB2"
    
    JOB3=$(sbatch --parsable eval/eval_embedding_d4_monash.slurm)
    echo "  Embedding D4 Monash: Job $JOB3"
    
    echo ""
fi

if [ "$FILTER" == "all" ] || [ "$FILTER" == "lotsa" ]; then
    echo "Submitting LOTSA in-distribution evaluations..."
    
    JOB4=$(sbatch --parsable eval/eval_standard_lotsa.slurm)
    echo "  Standard LOTSA: Job $JOB4"
    
    JOB5=$(sbatch --parsable eval/eval_embedding_d3_lotsa.slurm)
    echo "  Embedding D3 LOTSA: Job $JOB5"
    
    JOB6=$(sbatch --parsable eval/eval_embedding_d4_lotsa.slurm)
    echo "  Embedding D4 LOTSA: Job $JOB6"
    
    echo ""
fi

echo "=========================================="
echo "All jobs submitted!"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: uni2ts/eval-runs/"
echo "=========================================="
