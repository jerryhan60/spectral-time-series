#!/bin/bash
#SBATCH --job-name=ev_100kc
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_100kc_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_100kc_%j.err

# Eval 100K training curve: BL at 20K and 30K steps
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== 100K Training Curve Eval ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for STEP in epoch_199-step_20000 epoch_299-step_30000; do
  ckpt="${BASE}/baseline_m2d_s0_100k/checkpoints/${STEP}.ckpt"
  echo -e "\n--- BL_100k @ ${STEP} ---\nTime: $(date)"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== 100K curve eval complete: $(date) ==="
