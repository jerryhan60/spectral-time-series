#!/bin/bash
#SBATCH --job-name=ev_ctx
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_ctx_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_ctx_%j.err

# Context length ablation: evaluate BL and HD10 (m2d seed 0) at ctx=512, 1000, 2000
# (ctx=4000 already done). Tests cross-patch information leakage hypothesis.
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== Context Length Ablation (BL + HD10, m2d seed 0) ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for CTX in 512 1000 2000; do
  for MODEL in baseline_m2d_seed0_10k hd10_m2d_seed0_10k; do
    ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
    echo -e "\n--- ${MODEL} ctx=${CTX} ---\nTime: $(date)"
    if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
      --checkpoint "$ckpt" --context-length $CTX --batch-size 64
  done
done

echo "=== Context ablation complete: $(date) ==="
