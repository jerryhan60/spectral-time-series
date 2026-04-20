#!/bin/bash
#SBATCH --job-name=ev_ms70
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_ms70_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_ms70_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HF_HUB_OFFLINE=1

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/mshd10_m2d_s0_100k/checkpoints

for STEP in 70000 80000; do
  EPOCH=$(( STEP / 100 - 1 ))
  CKPT="${BASE}/epoch_${EPOCH}-step_${STEP}.ckpt"
  if [ -f "$CKPT" ]; then
    echo -e "\n--- MSHD10 @ ${STEP} steps ---"
    echo "Time: $(date)"
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
      --checkpoint "$CKPT" --context-length 4000 --batch-size 64
  else
    echo "Checkpoint $CKPT not found, skipping"
  fi
done

echo "=== Done: $(date) ==="
