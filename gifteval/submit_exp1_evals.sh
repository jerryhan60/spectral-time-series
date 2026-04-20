#!/bin/bash
# Submit GIFT-Eval jobs for all EXP-1a and EXP-1b checkpoints
# context_length=4000 (matching AK's eval setup)

set -euo pipefail

BASE="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted"
SLURM="/scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.slurm"
LOGDIR="/scratch/gpfs/EHAZAN/jh1161/logs"

echo "=== Submitting GIFT-Eval jobs (context_length=4000) ==="
echo ""

# --- EXP-1a: Baseline + Chebyshev Degree Sweep ---
echo "--- EXP-1a: Degree Sweep ---"

# Corrected baseline (zscore=8.0)
CKPT="${BASE}/m2_baseline_20260209_114203/checkpoints/epoch_999-step_100000.ckpt"
echo "Baseline (corrected): $CKPT"
sbatch --job-name=ge_baseline \
  --output="${LOGDIR}/ge_baseline_%j.out" \
  --error="${LOGDIR}/ge_baseline_%j.err" \
  --export=ALL,CHECKPOINT="$CKPT" \
  "$SLURM"

for DEG in 1 2 3 4 5 6 7; do
  case $DEG in
    1) DIR="m2_precond_d1_cheb_20260208_194538" ;;
    2) DIR="m2_precond_d2_cheb_20260208_195702" ;;
    3) DIR="m2_precond_d3_cheb_20260208_195919" ;;
    4) DIR="m2_precond_d4_cheb_20260208_200423" ;;
    5) DIR="m2_precond_d5_cheb_20260208_200423" ;;
    6) DIR="m2_precond_d6_cheb_20260208_200423" ;;
    7) DIR="m2_precond_d7_cheb_20260208_211558" ;;
  esac
  CKPT="${BASE}/${DIR}/checkpoints/epoch_999-step_100000.ckpt"
  echo "d=${DEG}: $CKPT"
  sbatch --job-name="ge_d${DEG}" \
    --output="${LOGDIR}/ge_d${DEG}_%j.out" \
    --error="${LOGDIR}/ge_d${DEG}_%j.err" \
    --export=ALL,CHECKPOINT="$CKPT" \
    "$SLURM"
done

echo ""
echo "--- EXP-1b: d=4 Lambda Sweep ---"

# Custom 4-tap
CKPT="${BASE}/m2_precond_d4_custom_20260209_111608/checkpoints/epoch_999-step_100000.ckpt"
echo "4tap custom: $CKPT"
sbatch --job-name=ge_4tap \
  --output="${LOGDIR}/ge_4tap_%j.out" \
  --error="${LOGDIR}/ge_4tap_%j.err" \
  --export=ALL,CHECKPOINT="$CKPT" \
  "$SLURM"

# Lambda sweep in the 112351 directory (versioned checkpoints)
SWEEP_DIR="${BASE}/m2_precond_d4_custom_20260209_112351/checkpoints"

# lam=0.25 (base checkpoint)
CKPT="${SWEEP_DIR}/epoch_999-step_100000.ckpt"
echo "lam=0.25: $CKPT"
sbatch --job-name=ge_lam025 \
  --output="${LOGDIR}/ge_lam025_%j.out" \
  --error="${LOGDIR}/ge_lam025_%j.err" \
  --export=ALL,CHECKPOINT="$CKPT" \
  "$SLURM"

# lam=1.0 (v1)
CKPT="${SWEEP_DIR}/epoch_999-step_100000-v1.ckpt"
echo "lam=1.0: $CKPT"
sbatch --job-name=ge_lam1 \
  --output="${LOGDIR}/ge_lam1_%j.out" \
  --error="${LOGDIR}/ge_lam1_%j.err" \
  --export=ALL,CHECKPOINT="$CKPT" \
  "$SLURM"

# lam=2.0 (v2)
CKPT="${SWEEP_DIR}/epoch_999-step_100000-v2.ckpt"
echo "lam=2.0: $CKPT"
sbatch --job-name=ge_lam2 \
  --output="${LOGDIR}/ge_lam2_%j.out" \
  --error="${LOGDIR}/ge_lam2_%j.err" \
  --export=ALL,CHECKPOINT="$CKPT" \
  "$SLURM"

# lam=0.5 (v3)
CKPT="${SWEEP_DIR}/epoch_999-step_100000-v3.ckpt"
echo "lam=0.5: $CKPT"
sbatch --job-name=ge_lam05 \
  --output="${LOGDIR}/ge_lam05_%j.out" \
  --error="${LOGDIR}/ge_lam05_%j.err" \
  --export=ALL,CHECKPOINT="$CKPT" \
  "$SLURM"

# lam=3.0 (v4)
CKPT="${SWEEP_DIR}/epoch_999-step_100000-v4.ckpt"
echo "lam=3.0: $CKPT"
sbatch --job-name=ge_lam3 \
  --output="${LOGDIR}/ge_lam3_%j.out" \
  --error="${LOGDIR}/ge_lam3_%j.err" \
  --export=ALL,CHECKPOINT="$CKPT" \
  "$SLURM"

# lam=10.0
CKPT="${BASE}/m2_precond_d4_custom_20260209_113038/checkpoints/epoch_999-step_100000.ckpt"
echo "lam=10.0: $CKPT"
sbatch --job-name=ge_lam10 \
  --output="${LOGDIR}/ge_lam10_%j.out" \
  --error="${LOGDIR}/ge_lam10_%j.err" \
  --export=ALL,CHECKPOINT="$CKPT" \
  "$SLURM"

echo ""
echo "=== All jobs submitted ==="
echo "Total: 8 (EXP-1a) + 7 (EXP-1b) = 15 jobs"
