#!/bin/bash
#SBATCH --job-name=eval_nabl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_nabl_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_nabl_%j.err
#SBATCH --dependency=afterok:5317047:5317048:5317702:5317717

# Evaluate: zero hints, random hints, stride=1, stride=32
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

OUTPUTS=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain

echo "=== Evaluating New Ablation Experiments ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for run_name in zero_hint_ms46_10k random_hint_ms46_10k hint_d6_s1_10k hint_d6_s32_10k; do
    # Search in both flat and nested output directory structures
    CKPT=$(ls -t ${OUTPUTS}/${run_name}*/checkpoints/*.ckpt 2>/dev/null | head -n1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -t ${OUTPUTS}/moirai2_small/lotsa_v1_unweighted/${run_name}*/checkpoints/*.ckpt 2>/dev/null | head -n1)
    fi
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
