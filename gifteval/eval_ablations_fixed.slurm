#!/bin/bash
#SBATCH --job-name=eval_abl2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=6:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_abl2_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_abl2_%j.err

# Evaluate ablation checkpoints + bl1k_100k@90K on GIFT-Eval (97 configs)
# Fixed path bug from original eval_abl script
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

OUTPUTS=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted

echo "=== Evaluating Ablation & Baseline Checkpoints ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Learned 4-tap @ 10K
CKPT="${OUTPUTS}/ablation_learned4_10k/checkpoints/epoch_99-step_10000.ckpt"
echo ""; echo "--- Evaluating: ablation_learned4_10k @ 10K ---"
echo "Checkpoint: $CKPT"; echo "Time: $(date)"
python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: ablation_learned4_10k"

# 2. Learned 16-tap @ 10K
CKPT="${OUTPUTS}/ablation_learned16_10k/checkpoints/epoch_99-step_10000.ckpt"
echo ""; echo "--- Evaluating: ablation_learned16_10k @ 10K ---"
echo "Checkpoint: $CKPT"; echo "Time: $(date)"
python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: ablation_learned16_10k"

# 3. Duplicate input @ 9K (timed out before 10K)
CKPT="${OUTPUTS}/ablation_duplicate_10k/checkpoints/epoch_89-step_9000.ckpt"
echo ""; echo "--- Evaluating: ablation_duplicate_10k @ 9K ---"
echo "Checkpoint: $CKPT"; echo "Time: $(date)"
python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: ablation_duplicate_10k"

# 4. Baseline 1K warmup @ 90K
CKPT="${OUTPUTS}/baseline_1kwarm_100k_20260228_063343/checkpoints/epoch_899-step_90000.ckpt"
echo ""; echo "--- Evaluating: bl1k_100k @ 90K ---"
echo "Checkpoint: $CKPT"; echo "Time: $(date)"
python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: bl1k_100k@90K"

echo ""
echo "=== All evaluations complete: $(date) ==="
