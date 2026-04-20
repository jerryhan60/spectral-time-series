#!/bin/bash
#SBATCH --job-name=test_variants
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/test_variants_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/test_variants_%j.err

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate

set -a
source .env
set +a

echo "=== GPU Training Test for All STU Variants ==="
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python tests/test_gpu_variants.py

echo ""
echo "=== Test Complete ==="
echo "End time: $(date)"
