#!/bin/bash
#SBATCH --job-name=ctx_swp
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ctx_swp_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ctx_swp_%j.err

# Context length sweep: evaluate HD10_m2d_seed0 and baseline_m2d_seed0
# at context lengths 500, 1000, 2000, 3000 (we already have 4000)
# Tests whether hint improvement is robust to context length
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
HD10="${BASE}/hd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
BL="${BASE}/baseline_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"

echo "=== Context Length Sweep ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for CTX in 500 1000 2000 3000; do
  echo -e "\n--- HD10 ctx=${CTX} ---"
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$HD10" --context-length $CTX --batch-size 64

  echo -e "\n--- Baseline ctx=${CTX} ---"
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$BL" --context-length $CTX --batch-size 64
done

echo "=== Context sweep complete: $(date) ==="
