#!/bin/bash
#SBATCH --job-name=eval_allseed
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=16:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_allseed_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_allseed_%j.err

# Evaluate ALL seed experiments for confidence intervals
# ms46: seeds 0, 1, 44(seed2), 7, 13 + original seed=42
# baseline: seeds 0, 1, 44(seed2), 7, 13 + original seed=42
# duplicate: seed=42 (original), 7, 13
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== Evaluating All Seeds for Confidence Intervals ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted
FAILED=0

for model in \
  ms46_seed0_10k ms46_seed1_10k ms46_seed2_10k ms46_seed7_10k ms46_seed13_10k \
  baseline_seed0_10k baseline_seed1_10k baseline_seed2_10k baseline_seed7_10k baseline_seed13_10k \
  ablation_duplicate_10k ablation_duplicate_seed7_10k ablation_duplicate_seed13_10k; do

  ckpt="${BASE}/${model}/checkpoints/epoch_99-step_10000.ckpt"
  echo ""
  echo "--- Evaluating ${model} ---"
  echo "Time: $(date)"
  if [ ! -f "$ckpt" ]; then
    echo "SKIP: checkpoint not found at $ckpt"
    continue
  fi
  python gifteval/eval_gifteval.py --checkpoint "$ckpt" --context-length 4000 --batch-size 64 || { echo "FAILED: $model"; FAILED=$((FAILED+1)); }
done

echo ""
echo "=== All seed evaluations complete: $(date) ==="
echo "Failures: $FAILED"
