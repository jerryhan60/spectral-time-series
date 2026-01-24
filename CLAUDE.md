# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Research repository for time series forecasting built on **Uni2TS** (Salesforce's universal time series forecasting library).

**Key Research Paper**: Woo et al. (2024). Unified Training of Universal Time Series Forecasting Transformers. ICML 2024

## Environment Setup (Princeton PLI Cluster)

```bash
# Module loads (required every session)
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate virtual environment
source uni2ts/venv/bin/activate

# Request GPU (interactive)
salloc --nodes=1 --ntasks=1 --mem=128G --time=03:01:00 --gres=gpu:1 --partition=pli --account=eladgroup
```

**Available accounts**: `eladgroup` (pli-low), `hazan_intern` (pli-low), `spectralssmtorch` (pli-low), `ehazan` (various gpu queues)

**Environment variables** (in `uni2ts/.env`):
- `LOTSA_V1_PATH`: Path to LOTSA dataset
- `LSF_PATH`: Path to Long Sequence Forecasting benchmark datasets

## Core Architecture

```
.
├── uni2ts/                          # Main framework code
│   ├── cli/                         # Command-line scripts
│   │   ├── train.py                 # Training (Hydra-based)
│   │   └── eval.py                  # Standard evaluation
│   ├── src/uni2ts/
│   │   ├── model/moirai/            # Moirai model implementations
│   │   ├── module/                  # Neural network modules
│   │   ├── transform/               # Data transformations
│   │   ├── data/                    # Data loading
│   │   └── loss/                    # Loss functions
│   └── cli/conf/                    # Hydra configurations
├── pretraining/                     # SLURM pretraining scripts
├── eval/                            # Evaluation scripts and SLURM jobs
├── eval_confs/                      # Dataset configurations (forecast_datasets.xlsx)
└── logs/                            # SLURM job outputs
```

## Common Commands

### Pre-training

```bash
# Standard pretraining
sbatch pretraining/pretrain_moirai.slurm
```

### Evaluation

```bash
# Standard evaluation (official Moirai)
sbatch eval/eval_comprehensive.slurm

# Evaluate custom checkpoint
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt eval/eval_comprehensive.slurm
```

### Interactive Training/Evaluation

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Training
python -m cli.train -cp conf/pretrain run_name=my_run model=moirai_small data=lotsa_v1_unweighted

# Evaluation
python -m cli.eval run_name=eval model=moirai_1.0_R_small model.patch_size=32 model.context_length=1000 data=lsf_test data.dataset_name=ETTh1 data.prediction_length=96
```

### SLURM Job Management

```bash
squeue -u $USER              # Check queue
tail -f logs/pretrain_*.out  # Monitor log
scancel JOBID                # Cancel job
```

## Key Configuration Parameters (Hydra)

- `model`: moirai_small, moirai_base, moirai_large
- `model.patch_size`: 8, 16, 32, 64, or 128
- `model.context_length`: Input context window size
- `data.prediction_length`: Forecast horizon

## Output Locations

- **Training**: `uni2ts/outputs/<run_name>/` (checkpoints, TensorBoard logs)
- **Evaluation**: `eval/outputs/` (metrics CSV, per-dataset outputs)
- **SLURM logs**: `logs/`

## Patch Size Configuration

Based on the Moirai paper (Appendix B.1), patch sizes are **frequency-dependent**:

| Frequency | Patch Size |
|-----------|------------|
| Quarterly (Q) | 8 |
| All others | 32 |

## Known Issues

**NaN handling**: Some datasets (e.g., Rideshare) contain NaN values. Evaluation scripts handle these and report status in results CSV.

**Memory**: Reduce `BATCH_SIZE` env var for OOM errors.

**Offline mode**: Evaluation scripts set `HF_HUB_OFFLINE=1`. Ensure datasets are cached locally.

## Documentation References

- `eval/README.md`: Evaluation script details
- `uni2ts/README.md`: Upstream Uni2TS documentation
