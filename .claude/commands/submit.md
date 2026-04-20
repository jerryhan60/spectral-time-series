Submit a new Moirai2 experiment with pre-flight validation. The user will describe what to submit as $ARGUMENTS.

**Pre-flight (MANDATORY):**
1. Check `uni2ts/cli/conf/pretrain/model/moirai2_small.yaml` — existing keys use `model.key=val`, NEW keys use `+model.key=val`
2. Verify SLURM partition: `pli` (eladgroup, pli-low) preferred, `ailab` (ehazan) backup. NEVER `gpu` or `grace`.
3. Always `--cpus-per-task=8`

**SLURM template:**
```
#!/bin/bash
#SBATCH --job-name=<name> --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/<name>_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/<name>_%j.err
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a && export HYDRA_FULL_ERROR=1
COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 seed=0"
python -m cli.train -cp conf/pretrain run_name=<name> $COMMON <OVERRIDES>
cd /scratch/gpfs/EHAZAN/jh1161
python gifteval/eval_gifteval.py --checkpoint "uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/<name>/checkpoints/epoch_99-step_10000.ckpt" --context-length 4000 --batch-size 64
```

After submission: record job ID in `slurm_job_log.md`, submit backup on alternate partition, report to user.
