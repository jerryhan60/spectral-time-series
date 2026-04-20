#!/bin/bash
#SBATCH --job-name=off_100k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/off_100k_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/off_100k_%A_%a.err
#SBATCH --array=0-5

# Matched-official 100K: BL and HD10 with 10K warmup (matching official Moirai 2.0 protocol)
# Array: 0-2 = BL seeds 0,2,7; 3-5 = HD10 seeds 0,2,7

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

SEEDS=(0 2 7 0 2 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

if [ $SLURM_ARRAY_TASK_ID -lt 3 ]; then
    METHOD="baseline"
    RUN_NAME="bl_official_s${SEED}_100k"
    HINT_ARGS=""
else
    METHOD="hd10"
    RUN_NAME="hd10_official_s${SEED}_100k"
    HINT_ARGS="model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 model.module_kwargs.hint_dropout=0.1"
fi

echo "========================================"
echo "Matched-Official 100K Training"
echo "Method: $METHOD, Seed: $SEED"
echo "Run name: $RUN_NAME"
echo "Key: 10K warmup (matching official Moirai 2.0)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "========================================"

python -m cli.train -cp conf/pretrain \
    run_name=$RUN_NAME \
    model=moirai2_small \
    data=lotsa_v1_moirai2 \
    trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    model.lr=1e-3 \
    model.weight_decay=0.1 \
    model.beta1=0.9 \
    model.beta2=0.98 \
    model.num_warmup_steps=10000 \
    model.anomaly_variance_ratio_threshold=0.0 \
    trainer.precision=bf16-mixed \
    tf32=false \
    +seed_everything=$SEED \
    $HINT_ARGS

echo "Training done: $(date)"

# Auto-eval after training
CKPT_DIR="outputs/pretrain/moirai2_small/lotsa_v1_moirai2/${RUN_NAME}/checkpoints"
CKPT=$(ls -t ${CKPT_DIR}/epoch_999-step_100000*.ckpt 2>/dev/null | head -1)

if [ -n "$CKPT" ]; then
    echo "Evaluating checkpoint: $CKPT"
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
        --checkpoint "$CKPT" \
        --context-length 4000 \
        --model "${RUN_NAME}"
    echo "Eval done: $(date)"
else
    echo "WARNING: No checkpoint found at step 100000"
fi
