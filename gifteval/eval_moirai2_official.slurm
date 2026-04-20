#!/bin/bash
#SBATCH --job-name=ge_m2_hf
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=2:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_m2_hf_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_m2_hf_%j.err

# GIFT-Eval evaluation of OFFICIAL Salesforce/moirai-2.0-R-small from HuggingFace
# This gives us the apples-to-apples published baseline for comparison
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a
export HF_HUB_OFFLINE=1

echo "=== GIFT-Eval: Official Moirai 2.0 Small (HuggingFace) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"

python gifteval/eval_gifteval.py \
  --model moirai-2.0-R-small \
  --context-length 4000 \
  --patch-size 32 \
  --batch-size 64

echo "=== Evaluation Complete ==="
echo "Results in: /scratch/gpfs/EHAZAN/jh1161/gifteval/results/"
