#!/bin/bash
#SBATCH --job-name=eval_abl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_abl_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_abl_%j.err

# Evaluate all 3 hint ablation experiments on GIFT-Eval (97 configs)
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

OUTPUTS=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain

echo "=== Evaluating Hint Ablation Experiments ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for run_name in ablation_duplicate_10k ablation_learned4_10k ablation_learned16_10k; do
    CKPT=$(ls -t ${OUTPUTS}/${run_name}*/checkpoints/*.ckpt 2>/dev/null | head -n1)
    if [ -n "$CKPT" ]; then
        echo ""
        echo "--- Evaluating: $run_name ---"
        echo "Checkpoint: $CKPT"
        echo "Time: $(date)"
        python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: $run_name"
    else
        echo "SKIP: No checkpoint found for $run_name"
    fi
done

echo ""
echo "=== All evaluations complete: $(date) ==="
echo "Results in: /scratch/gpfs/EHAZAN/jh1161/gifteval/results/"
