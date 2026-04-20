#!/bin/bash
#SBATCH --job-name=ev_tcurve
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=08:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_tcurve_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_tcurve_%j.err

# Training curve: evaluate BL and HD10 at steps 1K, 3K, 5K (m2d seed 0)
# Final (10K) already done. Shows when hint benefit emerges.
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== Training Curve Eval (BL + HD10, m2d seed 0) ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for STEP in epoch_9-step_1000 epoch_29-step_3000 epoch_49-step_5000; do
  for MODEL in baseline_m2d_seed0_10k hd10_m2d_seed0_10k; do
    ckpt="${BASE}/${MODEL}/checkpoints/${STEP}.ckpt"
    echo -e "\n--- ${MODEL} @ ${STEP} ---\nTime: $(date)"
    if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
      --checkpoint "$ckpt" --context-length 4000 --batch-size 64
  done
done

echo "=== Training curve eval complete: $(date) ==="
