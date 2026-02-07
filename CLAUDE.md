# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General behavioral guidelines
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


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
Use either one of these depending on which is full / not
salloc --nodes=1 --ntasks=1 --mem=128G --time=03:01:00 --gres=gpu:1 --partition=pli --account=eladgroup
salloc --nodes=1 --ntasks=1 --mem=128G --time=03:01:00 --gres=gpu:1 --partition=della --account=ehazan
salloc --nodes=1 --ntasks=1 --mem=128G --time=03:01:00 --gres=gpu:1 --partition=ailab --account=ehazan
```

**Available accounts**: `eladgroup` (pli-low), `hazan_intern` (pli-low), `spectralssmtorch` (pli-low), `ehazan` (various gpu queues)

**Environment variables** (in `uni2ts/.env`):
- `LOTSA_V1_PATH`: Path to LOTSA dataset
- `LSF_PATH`: Path to Long Sequence Forecasting benchmark datasets
- `GIFT_EVAL`: Path to GIFT-Eval benchmark datasets

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

**Best Practice**: Before submitting full training runs, always test on a small dataset first to catch configuration errors early:

```bash
# Quick validation test (CPU, no GPU needed)
python -m cli.train -cp conf/pretrain \
  run_name=test_validation \
  model=moirai_small_stu \
  model.num_warmup_steps=1 \
  data=test_small \
  trainer.max_epochs=1 \
  train_dataloader.num_batches_per_epoch=3 \
  train_dataloader.batch_size=4 \
  trainer.accelerator=cpu

# Or interactive GPU test (faster, catches GPU-specific issues)
salloc --partition=ailab --account=ehazan --gres=gpu:1 --time=00:30:00 --mem=32G
python -m cli.train -cp conf/pretrain \
  run_name=test_gpu \
  model=moirai_small_stu \
  model.num_warmup_steps=1 \
  data=test_small \
  trainer.max_epochs=2 \
  train_dataloader.num_batches_per_epoch=5 \
  train_dataloader.batch_size=8
```

```bash
# Standard pretraining (after validation passes)
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

## Flash STU (Spectral Transform Units)

Location: `/scratch/gpfs/EHAZAN/jh1161/flash-stu-2/`

**Paper**: Flash STU: Fast Spectral Transform Units (arXiv:2409.10489)

### Installation Status

- **flash-stu**: Installed (editable mode)
- **flash-attn**: Installed
- **flash-fft-conv**: NOT installed (compilation issues on cluster)

### Required Configuration

Since Flash FFT Conv is not installed, always use these settings:

```python
from flash_stu import FlashSTU, FlashSTUConfig

config = FlashSTUConfig(
    n_embd=512,
    n_layers=12,
    n_heads=8,
    seq_len=2048,
    vocab_size=50257,
    use_flash_fft=False,  # REQUIRED: Flash FFT Conv not installed
    use_attn=True,        # Hybrid STU + Attention (flash-attn installed)
)

model = FlashSTU(config).cuda()
```

### STU-Only Mode (No Attention)

```python
config = FlashSTUConfig(
    n_embd=512,
    n_layers=12,
    n_heads=8,
    seq_len=2048,
    vocab_size=50257,
    use_flash_fft=False,
    use_attn=False,       # Pure STU, no attention layers
    use_cache=False,      # Disable KV cache for STU-only
)
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_embd` | Hidden dimension | 1536 |
| `n_layers` | Total layers (STU + Attention alternate) | 26 |
| `num_eigh` | Number of spectral filters | 24 |
| `use_flash_fft` | Use Flash FFT Conv (set False) | True |
| `use_attn` | Include attention layers | True |
| `use_approx` | Approx mode (~50x fewer STU params) | True |

### Installing Flash FFT Conv (Future)

Requires GPU node with internet access. Clone repo on login node first:
```bash
# Login node:
git clone https://github.com/HazyResearch/flash-fft-conv.git /scratch/gpfs/EHAZAN/jh1161/flash-fft-conv

# GPU node:
MAX_JOBS=4 pip install /scratch/gpfs/EHAZAN/jh1161/flash-fft-conv/csrc/flashfftconv
pip install /scratch/gpfs/EHAZAN/jh1161/flash-fft-conv
```

## GIFT-Eval Benchmark

Location: `/scratch/gpfs/EHAZAN/jh1161/gifteval/`

**Paper**: GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (arXiv:2410.10393)

### Initial Setup (run ONCE on login node)

```bash
bash /scratch/gpfs/EHAZAN/jh1161/gifteval/setup_gifteval.sh
```

### Quick Evaluation (~30 min, 8 datasets)

```bash
# Evaluate a checkpoint
sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval_quick.slurm

# Or use HuggingFace model
sbatch --export=MODEL=moirai-1.1-R-small /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval_quick.slurm
```

### Full Benchmark (~8-12 hours, 98 configurations)

```bash
sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.slurm
```

### Interactive (from GPU node)

```bash
./gifteval/eval_interactive.sh /path/to/checkpoint.ckpt        # Quick
./gifteval/eval_interactive.sh /path/to/checkpoint.ckpt --full # Full
```

### Output

Results saved to `/scratch/gpfs/EHAZAN/jh1161/gifteval/results/`:
- `gifteval_results_<model>_<timestamp>.csv` - All metrics
- `all_results_<model>.csv` - Leaderboard format

## Documentation References

- `eval/README.md`: Evaluation script details
- `uni2ts/README.md`: Upstream Uni2TS documentation
- `flash-stu-2/README.md`: Flash STU documentation
- `gifteval/README.md`: GIFT-Eval benchmark setup and usage


## SLURM Job Log

Job logs are maintained in a separate file: `/scratch/gpfs/EHAZAN/jh1161/slurm_job_log.md`

Update that file after submitting SLURM jobs to track:
- Trained model checkpoints and configs
- Active/recent jobs with status
- Cancelled/failed jobs
- Evaluation results