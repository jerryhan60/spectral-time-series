#!/bin/bash
#SBATCH --job-name=eval_addl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_addl_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_addl_%j.err
#SBATCH --array=0-5

# Evaluate BL and HD10 (seed 0, 10K) on FEV-Bench and LSF benchmarks
# Array: 0-2 = FEV-bench (official, BL, HD10), 3-5 = LSF (official, BL, HD10)

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

export HF_DATASETS_CACHE="/scratch/gpfs/EHAZAN/jh1161/gifteval/.hf_cache"
export HF_HOME="/scratch/gpfs/EHAZAN/jh1161/gifteval/.hf_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

BASE="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2"
BL_CKPT="${BASE}/baseline_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
HD10_CKPT="${BASE}/hd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
OFFICIAL="moirai-2.0-R-small"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/scratch/gpfs/EHAZAN/jh1161/gifteval/results"

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "=== FEV-Bench: Official Moirai 2.0 ==="
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/fev_bench_moirai2.py \
      --ckpt "$OFFICIAL" \
      --output "${RESULTS_DIR}/fev_bench_official_${TIMESTAMP}.csv" \
      --model-name "official_moirai2" \
      --context-length 4000 --batch-size 512 \
      --tasks-yaml /scratch/gpfs/EHAZAN/jh1161/gifteval/fev_bench_tasks.yaml
    ;;
  1)
    echo "=== FEV-Bench: Baseline seed 0 ==="
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/fev_bench_moirai2.py \
      --ckpt "$BL_CKPT" \
      --output "${RESULTS_DIR}/fev_bench_bl_s0_${TIMESTAMP}.csv" \
      --model-name "bl_s0_10k" \
      --context-length 4000 --batch-size 512 \
      --tasks-yaml /scratch/gpfs/EHAZAN/jh1161/gifteval/fev_bench_tasks.yaml
    ;;
  2)
    echo "=== FEV-Bench: HD10 seed 0 ==="
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/fev_bench_moirai2.py \
      --ckpt "$HD10_CKPT" \
      --output "${RESULTS_DIR}/fev_bench_hd10_s0_${TIMESTAMP}.csv" \
      --model-name "hd10_s0_10k" \
      --context-length 4000 --batch-size 512 \
      --tasks-yaml /scratch/gpfs/EHAZAN/jh1161/gifteval/fev_bench_tasks.yaml
    ;;
  3)
    echo "=== LSF: Official Moirai 2.0 ==="
    for DATASET in ETTh1 ETTh2 ETTm1 ETTm2 weather electricity; do
      for PL in 96 192 336 720; do
        echo "--- $DATASET prediction_length=$PL ---"
        python -m cli.eval \
          run_name=lsf_official_${DATASET}_${PL} \
          model=moirai_1.0_R_small \
          model.patch_size=32 \
          model.context_length=4000 \
          data=lsf_test \
          data.dataset_name=$DATASET \
          data.prediction_length=$PL \
          data.mode=S 2>&1 | tail -5
      done
    done
    ;;
  4)
    echo "=== LSF: Baseline seed 0 ==="
    for DATASET in ETTh1 ETTh2 ETTm1 ETTm2 weather electricity; do
      for PL in 96 192 336 720; do
        echo "--- $DATASET prediction_length=$PL ---"
        python -m cli.eval \
          run_name=lsf_bl_s0_${DATASET}_${PL} \
          model=moirai2_small \
          model.patch_size=32 \
          model.context_length=4000 \
          +model.ckpt_path=$BL_CKPT \
          data=lsf_test \
          data.dataset_name=$DATASET \
          data.prediction_length=$PL \
          data.mode=S 2>&1 | tail -5
      done
    done
    ;;
  5)
    echo "=== LSF: HD10 seed 0 ==="
    for DATASET in ETTh1 ETTh2 ETTm1 ETTm2 weather electricity; do
      for PL in 96 192 336 720; do
        echo "--- $DATASET prediction_length=$PL ---"
        python -m cli.eval \
          run_name=lsf_hd10_s0_${DATASET}_${PL} \
          model=moirai2_small \
          model.patch_size=32 \
          model.context_length=4000 \
          +model.ckpt_path=$HD10_CKPT \
          data=lsf_test \
          data.dataset_name=$DATASET \
          data.prediction_length=$PL \
          data.mode=S 2>&1 | tail -5
      done
    done
    ;;
esac

echo "Done: $(date)"
