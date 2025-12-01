#!/bin/bash
#SBATCH --job-name=moirai_comprehensive_eval
#SBATCH --output=logs/eval_comprehensive_%j.out
#SBATCH --error=logs/eval_comprehensive_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=hazan_intern
#SBATCH --mail-type=END
#SBATCH --mail-user=jh1161@princeton.edu

# Print job information
echo "=========================================="
echo "Comprehensive Moirai Evaluation (Python Script)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Arguments: $@"
echo ""

# Change to project directory
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Load environment
echo "Activating virtual environment..."
source venv/bin/activate

# Force HuggingFace to use offline mode (skip online checks)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HuggingFace offline mode enabled"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Run the comprehensive evaluation script
# Pass all arguments from the slurm script to the python script
echo "Running comprehensive_evaluation.py..."
python /scratch/gpfs/EHAZAN/jh1161/eval/comprehensive_evaluation.py "$@"

echo ""
echo "=========================================="
echo "Job Completed"
echo "=========================================="
echo "End time: $(date)"
