#!/bin/bash
#SBATCH --job-name=eval_monash_cmp
#SBATCH --output=logs/eval_monash_comparison_%j.out
#SBATCH --error=logs/eval_monash_comparison_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=della
#SBATCH --account=ehazan

echo "=========================================="
echo "Monash Dataset Comparison: Baseline vs STU"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd /scratch/gpfs/EHAZAN/jh1161

# Load environment
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source uni2ts/venv/bin/activate

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run evaluation
python eval_monash_comparison.py

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
