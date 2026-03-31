#!/bin/bash
#SBATCH --job-name=ptst_ov
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ptst_ov_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ptst_ov_%A_%a.err
#SBATCH --array=0-9

# Overlapping patches baseline for PatchTST
# Tests if overlapping patches provide the same cross-patch info as hints
# 50% overlap (stride=8 with patch=16) doubles sequence length → ~4x attention cost
# 5 seeds x 2 datasets = 10 tasks

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/patchtst_hint

SEEDS=(0 1 2 7 42 0 1 2 7 42)
DATASETS=(ETTh1 ETTh1 ETTh1 ETTh1 ETTh1 ETTh2 ETTh2 ETTh2 ETTh2 ETTh2)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=== Overlap baseline: ${DATASET} seed=${SEED} ==="
echo "Job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | Time: $(date)"

# Non-overlapping baseline (stride=16, same as hint experiments)
echo -e "\n--- ${DATASET} stride=16 (no overlap) seed=${SEED} ---"
python train_eval_overlap.py \
  --dataset ${DATASET} --pred_len 96 --patch_stride 16 --seed ${SEED} --epochs 100

# 50% overlap (stride=8, doubles num_patches → ~4x attention)
echo -e "\n--- ${DATASET} stride=8 (50% overlap) seed=${SEED} ---"
python train_eval_overlap.py \
  --dataset ${DATASET} --pred_len 96 --patch_stride 8 --seed ${SEED} --epochs 100

echo "=== Complete: $(date) ==="
