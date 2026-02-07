#!/bin/bash
#SBATCH --job-name=validate_stu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/validate_stu_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/validate_stu_%j.err

# Load modules
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate virtual environment
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate

echo "=== STU Change Validation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

python validate_stu_change.py

echo ""
echo "=== Validation Complete ==="
