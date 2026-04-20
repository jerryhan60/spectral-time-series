#!/bin/bash
#SBATCH --job-name=ctx_hd10
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ctx_hd10_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ctx_hd10_%j.err

# Context length sweep for HD10 and baseline (m2d seed 0)
# Tests hint benefit at ctx=1000, 2000, 4000
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
HD10="${BASE}/hd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
BL="${BASE}/baseline_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"

echo "=== Context Length x HD10 Interaction ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for CTX in 1000 2000; do
  echo -e "\n--- HD10 @ ctx=${CTX} ---\nTime: $(date)"
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$HD10" --context-length "$CTX" --batch-size 64

  echo -e "\n--- Baseline @ ctx=${CTX} ---\nTime: $(date)"
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$BL" --context-length "$CTX" --batch-size 64
done

echo "=== Context sweep complete: $(date) ==="
