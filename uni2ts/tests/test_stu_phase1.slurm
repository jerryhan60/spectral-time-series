#!/bin/bash
#SBATCH --job-name=test_stu_phase1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=logs/test_stu_phase1_%j.out
#SBATCH --error=logs/test_stu_phase1_%j.err

# Load modules
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate virtual environment
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

echo "=== Environment Info ==="
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')"
echo ""

echo "=== Running Phase 1 STU Tests ==="
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python tests/test_stu_phase1.py

echo ""
echo "=== Test Complete ==="
