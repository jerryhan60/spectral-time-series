#!/bin/bash
#SBATCH --job-name=eval_std_monash
#SBATCH --output=logs/eval_std_monash_%j.out
#SBATCH --error=logs/eval_std_monash_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup

echo "Standard Baseline Evaluation on Monash Benchmark"
echo "================================================"

cd /scratch/gpfs/EHAZAN/jh1161
module load anaconda3/2024.6
module load cudatoolkit/12.6
source uni2ts/venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/checkpoints/last.ckpt"
OUTPUT_DIR="uni2ts/eval-runs/standard_monash"

python eval/comprehensive_evaluation.py \
    --mode standard \
    --model-path "$MODEL_PATH" \
    --dataset-source monash \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 32

echo "Results saved to: $OUTPUT_DIR"
