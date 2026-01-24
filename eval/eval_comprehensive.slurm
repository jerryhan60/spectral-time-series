#!/bin/bash
#SBATCH --job-name=eval_comprehensive
#SBATCH --output=logs/eval_comprehensive_%j.out
#SBATCH --error=logs/eval_comprehensive_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jh1161@princeton.edu

# ============================================================================
# Comprehensive Evaluation SLURM Script
# ============================================================================
#
# Usage:
#   sbatch eval/eval_comprehensive.slurm
#   sbatch --export=DATASET_SOURCE=monash,EVAL_MODE=standard eval/eval_comprehensive.slurm
#   sbatch --export=DATASET_SOURCE=lotsa-indist,EVAL_MODE=embedding,PRECOND_DEGREE=3 eval/eval_comprehensive.slurm
#
# Parameters:
#   DATASET_SOURCE: monash (default) or lotsa-indist
#   EVAL_MODE: standard, embedding (default: standard)
#   PRECOND_DEGREE: 3 or 4 (for embedding mode, default: 3)
#   MODEL_PATH: custom model path (optional, overrides default)
#
# ============================================================================

# Print job information
echo "=========================================="
echo "Comprehensive Model Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Change to project directory
cd /scratch/gpfs/EHAZAN/jh1161

# Load environment
echo "Loading environment..."
module load anaconda3/2024.6
module load cudatoolkit/12.6
source uni2ts/venv/bin/activate

# Set offline mode for HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# Configuration (can be overridden via --export)
# ============================================================================
DATASET_SOURCE=${DATASET_SOURCE:-monash}
EVAL_MODE=${EVAL_MODE:-standard}
PRECOND_DEGREE=${PRECOND_DEGREE:-3}

# Default model paths
BASELINE_MODEL="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/checkpoints/last.ckpt"
D3_EMBEDDING_MODEL="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d3_20251205_174706/checkpoints/last.ckpt"
D4_EMBEDDING_MODEL="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d4_20251205_174706/checkpoints/last.ckpt"

# Select model based on mode
if [ "$EVAL_MODE" == "standard" ]; then
    MODEL_PATH=${MODEL_PATH:-$BASELINE_MODEL}
    OUTPUT_DIR="uni2ts/eval-runs/standard_${DATASET_SOURCE}"
elif [ "$EVAL_MODE" == "embedding" ]; then
    if [ "$PRECOND_DEGREE" == "3" ]; then
        MODEL_PATH=${MODEL_PATH:-$D3_EMBEDDING_MODEL}
    else
        MODEL_PATH=${MODEL_PATH:-$D4_EMBEDDING_MODEL}
    fi
    OUTPUT_DIR="uni2ts/eval-runs/embedding_precond_d${PRECOND_DEGREE}_${DATASET_SOURCE}"
else
    echo "ERROR: Unknown EVAL_MODE: $EVAL_MODE"
    echo "Valid modes: standard, embedding"
    exit 1
fi

echo "=========================================="
echo "Evaluation Configuration:"
echo "=========================================="
echo "Dataset Source: $DATASET_SOURCE"
echo "Evaluation Mode: $EVAL_MODE"
if [ "$EVAL_MODE" == "embedding" ]; then
    echo "Precond Degree: $PRECOND_DEGREE"
fi
echo "Model Path: $MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Run Evaluation
# ============================================================================
echo "Starting evaluation..."
echo ""

cd /scratch/gpfs/EHAZAN/jh1161

if [ "$EVAL_MODE" == "standard" ]; then
    python eval/comprehensive_evaluation.py \
        --mode standard \
        --model-path "$MODEL_PATH" \
        --dataset-source "$DATASET_SOURCE" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 32
elif [ "$EVAL_MODE" == "embedding" ]; then
    python eval/comprehensive_evaluation.py \
        --mode embedding \
        --model-path "$MODEL_PATH" \
        --dataset-source "$DATASET_SOURCE" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 32
fi

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

exit $EXIT_CODE
