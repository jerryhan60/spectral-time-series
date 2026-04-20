#!/bin/bash
#SBATCH --job-name=ptst_fix
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ptst_fix_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ptst_fix_%A_%a.err
#SBATCH --array=0-9

# Test monic coefficient fix for PatchTST hint preconditioning
# 2 conditions x 5 seeds = 10 tasks
# Conditions: hint (monic fix), hint+normalize
# Run: sbatch --partition=pli --account=hazan_intern --qos=pli-low run_monic_fix.slurm

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/patchtst_hint

# Task layout: 5 seeds x 2 conditions
SEEDS=(0 1 2 7 42 0 1 2 7 42)
# First 5: hint (monic fix, no normalize)
# Last 5:  hint (monic fix + normalize)
NORMALIZE=(0 0 0 0 0 1 1 1 1 1)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
NORM=${NORMALIZE[$SLURM_ARRAY_TASK_ID]}

if [ "$NORM" = "1" ]; then
    NORM_FLAG="--hint_normalize"
    SUFFIX="hint_monic_norm"
else
    NORM_FLAG=""
    SUFFIX="hint_monic"
fi

echo "=== PatchTST monic fix: ${SUFFIX} seed=${SEED} ==="
echo "Job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | Time: $(date)"

for DATASET in ETTh1 ETTh2; do
  for PRED_LEN in 96 336; do
    echo -e "\n--- ${DATASET} pred=${PRED_LEN} ${SUFFIX} seed=${SEED} ---"
    python train_eval.py \
      --dataset ${DATASET} \
      --pred_len ${PRED_LEN} \
      --hint_mode hint \
      --seed ${SEED} \
      --epochs 100 \
      --lr 1e-3 \
      --batch_size 32 \
      ${NORM_FLAG} \
      --output_dir /scratch/gpfs/EHAZAN/jh1161/patchtst_hint/results_monic
  done
done

echo "=== Complete: $(date) ==="
