#!/bin/bash
#SBATCH --job-name=zh_eval
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/zh_eval_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/zh_eval_%j.err

# Mechanistic ablation: evaluate HD10_m2d_seed0 with hints zeroed at inference
# Compares with normal eval (hints active) to measure inference-time hint value
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

CKPT=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/hd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt

echo "=== Zero-Hints Ablation ==="
echo "Evaluating HD10_m2d_seed0 with --zero-hints flag"
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "$CKPT" --context-length 4000 --batch-size 64 --zero-hints

echo "=== Zero-hints eval complete: $(date) ==="
