#!/bin/bash
#SBATCH --job-name=s13_res
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/s13_res_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/s13_res_%j.err

# Resume ms46_seed13 from epoch 39 and eval when done
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_unweighted trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.time_precondition_degree=4 +model.module_kwargs.time_precondition_extra_hints=6:16"

echo "=== Resume ms46_seed13 + eval ==="
CKPT=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms46_seed13_10k/checkpoints/epoch_39-step_4000.ckpt

python -m cli.train -cp conf/pretrain run_name=ms46_seed13_10k $COMMON seed=13 $HINT \
  trainer.ckpt_path=$CKPT

# Eval
ckpt_final=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms46_seed13_10k/checkpoints/epoch_99-step_10000.ckpt
if [ -f "$ckpt_final" ]; then
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt_final" --context-length 4000 --batch-size 64
fi

echo "=== Done: $(date) ==="
