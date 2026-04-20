#!/bin/bash
#SBATCH --job-name=fev_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/gifteval/logs/fev_bench_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/gifteval/logs/fev_bench_%j.err

# FEV-Bench Evaluation for Moirai-2
# Usage:
#   CKPT_PATH=/path/to/ckpt MODEL_NAME=my_model sbatch eval_fev_bench.slurm

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

set -euo pipefail

# FEV datasets cache (copied from ak's cache)
export HF_DATASETS_CACHE="/scratch/gpfs/EHAZAN/jh1161/gifteval/.hf_cache"
export HF_HOME="/scratch/gpfs/EHAZAN/jh1161/gifteval/.hf_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

CKPT_PATH="${CKPT_PATH:?Must set CKPT_PATH}"
MODEL_NAME="${MODEL_NAME:-moirai2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_TASKS="${NUM_TASKS:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="/scratch/gpfs/EHAZAN/jh1161/gifteval/results/fev_bench_${MODEL_NAME}_${TIMESTAMP}.csv"

mkdir -p /scratch/gpfs/EHAZAN/jh1161/gifteval/logs
mkdir -p /scratch/gpfs/EHAZAN/jh1161/gifteval/results

echo "========================================"
echo "FEV-Bench Moirai-2 Evaluation"
echo "========================================"
echo "Checkpoint: $CKPT_PATH"
echo "Output: $OUTPUT_CSV"
echo "Model: $MODEL_NAME"
echo "Context Length: $CONTEXT_LENGTH"
echo "Batch Size: $BATCH_SIZE"
echo "Num Tasks: ${NUM_TASKS:-all 100}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo "========================================"

python /scratch/gpfs/EHAZAN/jh1161/gifteval/fev_bench_moirai2.py \
  --ckpt "$CKPT_PATH" \
  --output "$OUTPUT_CSV" \
  --model-name "$MODEL_NAME" \
  --context-length "$CONTEXT_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --tasks-yaml /scratch/gpfs/EHAZAN/jh1161/gifteval/fev_bench_tasks.yaml \
  ${NUM_TASKS:+--num-tasks "$NUM_TASKS"}

echo "========================================"
echo "FEV-Bench evaluation complete!"
echo "Results: $OUTPUT_CSV"
echo "End: $(date)"
echo "========================================"
