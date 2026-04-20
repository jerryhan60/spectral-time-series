#!/bin/bash
#SBATCH --job-name=ev_off
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_off_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_off_%j.err

# Evaluate the OFFICIAL Moirai2-Small from HuggingFace
# This gives us the published baseline to compare against.

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== Official Moirai2-Small Eval ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# Eval all 3 official models for calibration
for MODEL in Salesforce/moirai-2.0-R-small Salesforce/moirai-1.1-R-small Salesforce/moirai-1.0-R-small; do
  echo -e "\n--- ${MODEL} ---\nTime: $(date)"
  python gifteval/eval_gifteval.py \
    --model "${MODEL}" \
    --context-length 4000 \
    --batch-size 64
done

echo "=== Complete: $(date) ==="
