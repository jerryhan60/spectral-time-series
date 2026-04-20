#!/bin/bash
#SBATCH --job-name=profile_stu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/profile_stu_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/profile_stu_%j.err

# Load modules
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate virtual environment
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate

echo "=== STU Performance Profiling ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "Time: $(date)"
echo ""

python profile_stu.py

echo ""
echo "=== Profiling Complete ==="
echo "Time: $(date)"
