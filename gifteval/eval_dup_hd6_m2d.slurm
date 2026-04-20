#!/bin/bash
#SBATCH --job-name=ev_duph
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_duph_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_duph_%j.err

# Evaluate dup_hd10_m2d_s0 and hd6_m2d_s0 (training done, eval pending in deg_ema)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== Eval dup_hd10 + hd6 (m2d, seed 0) ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for MODEL in dup_hd10_m2d_seed0_10k hd6_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---\nTime: $(date)"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== dup_hd10 + hd6 eval complete: $(date) ==="
