#!/bin/bash
#SBATCH --job-name=ge_b100k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_b100k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_b100k_%j.err

# CRITICAL: Evaluate the PROPERLY CONFIGURED 100K baseline
# This run used anomaly_zscore_threshold=8.0 (same as our 10K experiments)
# The previous 100K baseline (MASE 1.2878) used zscore=0.0 and had massive loss spikes
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

CKPT="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_baseline_20260209_114203/checkpoints/epoch_999-step_100000.ckpt"

echo "=== GIFT-Eval: Proper 100K Baseline (zscore=8.0) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Time: $(date)"
echo ""
echo "Checkpoint: $CKPT"
echo "NOTE: This baseline had zscore=8.0, warmup=10K. Previous 100K baseline had zscore=0.0!"

python gifteval/eval_gifteval.py --checkpoint "$CKPT"

echo "=== Evaluation Complete ==="
echo "End: $(date)"
