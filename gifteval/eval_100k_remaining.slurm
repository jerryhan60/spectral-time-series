#!/bin/bash
#SBATCH --job-name=ev_100r
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_100r_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_100r_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HF_HUB_OFFLINE=1

echo "=== 100K Remaining Curve Eval ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

# BL checkpoints
for STEP in 50000 60000 70000 80000 90000 100000; do
  EPOCH=$((STEP/100 - 1))
  ckpt="${BASE}/baseline_m2d_s0_100k/checkpoints/epoch_${EPOCH}-step_${STEP}.ckpt"
  if [ -f "$ckpt" ]; then
    echo -e "\n--- BL @ ${STEP} steps ---\nTime: $(date)"
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
      --checkpoint "$ckpt" --context-length 4000 --batch-size 64
  fi
done

# HD10 checkpoints  
for STEP in 60000 70000 80000 90000 100000; do
  EPOCH=$((STEP/100 - 1))
  ckpt="${BASE}/hd10_m2d_s0_100k/checkpoints/epoch_${EPOCH}-step_${STEP}.ckpt"
  if [ -f "$ckpt" ]; then
    echo -e "\n--- HD10 @ ${STEP} steps ---\nTime: $(date)"
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
      --checkpoint "$ckpt" --context-length 4000 --batch-size 64
  fi
done

echo "=== 100K eval complete: $(date) ==="
