#!/bin/bash
#SBATCH --job-name=ev_ms100
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_ms100_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_ms100_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HF_HUB_OFFLINE=1

echo "=== MSHD10 100K Curve Eval ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

# MSHD10 checkpoints 10K-100K (skip any not yet saved)
for STEP in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000; do
  EPOCH=$((STEP/100 - 1))
  ckpt="${BASE}/mshd10_m2d_s0_100k/checkpoints/epoch_${EPOCH}-step_${STEP}.ckpt"
  if [ -f "$ckpt" ]; then
    echo -e "\n--- MSHD10 @ ${STEP} steps ---\nTime: $(date)"
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
      --checkpoint "$ckpt" --context-length 4000 --batch-size 64
  else
    echo "Checkpoint not yet available: epoch_${EPOCH}-step_${STEP}.ckpt"
  fi
done

echo "=== MSHD10 100K eval complete: $(date) ==="
