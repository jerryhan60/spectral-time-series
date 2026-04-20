#!/bin/bash
#SBATCH --job-name=ptst
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ptst_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ptst_%A_%a.err
#SBATCH --array=0-19

# PatchTST hint preconditioning experiment
# 4 modes x 5 seeds = 20 tasks
# Run: sbatch --partition=pli --account=hazan_intern --qos=pli-low run_patchtst.slurm

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/patchtst_hint

MODES=(none hint zero duplicate none hint zero duplicate none hint zero duplicate none hint zero duplicate none hint zero duplicate)
SEEDS=(0 0 0 0 1 1 1 1 2 2 2 2 7 7 7 7 42 42 42 42)

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== PatchTST: mode=${MODE} seed=${SEED} ==="
echo "Job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | Time: $(date)"

# Run on ETTh1 with pred_len=96 (standard benchmark)
for DATASET in ETTh1 ETTh2; do
  for PRED_LEN in 96 336; do
    echo -e "\n--- ${DATASET} pred=${PRED_LEN} mode=${MODE} seed=${SEED} ---"
    python train_eval.py \
      --dataset ${DATASET} \
      --pred_len ${PRED_LEN} \
      --hint_mode ${MODE} \
      --seed ${SEED} \
      --epochs 100 \
      --lr 1e-3 \
      --batch_size 32
  done
done

echo "=== Complete: $(date) ==="
