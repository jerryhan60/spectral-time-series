#!/bin/bash
#SBATCH --job-name=eval_emb_d2_monash_100k
#SBATCH --output=logs/eval_emb_d2_monash_100k_%j.out
#SBATCH --error=logs/eval_emb_d2_monash_100k_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup

echo "Embedding Precond D2 Evaluation on Monash Benchmark"
echo "===================================================="

cd /scratch/gpfs/EHAZAN/jh1161
module load anaconda3/2024.6
module load cudatoolkit/12.6
source uni2ts/venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d2_20251206_171408/checkpoints/epoch_999-step_100000.ckpt"
OUTPUT_DIR="uni2ts/eval-runs/embedding_precond_d2_monash_100k"

python eval/comprehensive_evaluation.py \
    --mode embedding \
    --model-path "$MODEL_PATH" \
    --dataset-source monash \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 32

echo "Results saved to: $OUTPUT_DIR"
