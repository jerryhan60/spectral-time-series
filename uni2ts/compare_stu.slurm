#!/bin/bash
#SBATCH --job-name=compare_stu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/compare_stu_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/compare_stu_%j.err

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate

echo "=== STU Training Comparison ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

python compare_stu_training.py

echo ""
echo "=== Comparison Complete ==="
