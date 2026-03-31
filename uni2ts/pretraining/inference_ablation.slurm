#!/bin/bash
#SBATCH --job-name=inf_abl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/inf_abl_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/inf_abl_%j.err

# Inference-time hint ablation: evaluate HD10 and MS46 with hints zeroed
# Tests whether the model USES hints at inference or benefits from training regularization only
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== Inference-Time Hint Ablation ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"
echo "Tests: HD10 and MS46 with --zero-hints at eval time"

# HD10 m2d seed 0 (best result: -4.99%) — with hints zeroed
ckpt="${BASE}/hd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
echo -e "\n--- HD10_m2d_s0 with zero hints ---\nTime: $(date)"
if [ -f "$ckpt" ]; then
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64 --zero-hints
else
  echo "SKIP: $ckpt not found"
fi

# HD10 m2d seed 2 (second best: -2.77%) — with hints zeroed
ckpt="${BASE}/hd10_m2d_seed2_10k/checkpoints/epoch_99-step_10000.ckpt"
echo -e "\n--- HD10_m2d_s2 with zero hints ---\nTime: $(date)"
if [ -f "$ckpt" ]; then
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64 --zero-hints
else
  echo "SKIP: $ckpt not found"
fi

# MS46 m2d seed 0 — with hints zeroed
ckpt="${BASE}/ms46_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
echo -e "\n--- MS46_m2d_s0 with zero hints ---\nTime: $(date)"
if [ -f "$ckpt" ]; then
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64 --zero-hints
else
  echo "SKIP: $ckpt not found"
fi

# MS46 m2d seed 2 (best: -3.48%) — with hints zeroed
ckpt="${BASE}/ms46_m2d_seed2_10k/checkpoints/epoch_99-step_10000.ckpt"
echo -e "\n--- MS46_m2d_s2 with zero hints ---\nTime: $(date)"
if [ -f "$ckpt" ]; then
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64 --zero-hints
else
  echo "SKIP: $ckpt not found"
fi

echo "=== Inference ablation complete: $(date) ==="
