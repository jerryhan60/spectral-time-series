#!/bin/bash
#SBATCH --job-name=ctx_8k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ctx_8k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ctx_8k_%j.err

# Context length scaling: HD10 and baseline at ctx=8000
# Tests whether hint benefit persists/diminishes/grows with longer context
# Already have: ctx=1000 (running), ctx=4000 (done)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
HD10="${BASE}/hd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
BL="${BASE}/baseline_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"

echo "=== Context 8K: HD10 vs Baseline ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

echo -e "\n--- HD10 @ ctx=8000 ---\nTime: $(date)"
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "$HD10" --context-length 8000 --batch-size 32

echo -e "\n--- Baseline @ ctx=8000 ---\nTime: $(date)"
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "$BL" --context-length 8000 --batch-size 32

echo "=== Context 8K complete: $(date) ==="
