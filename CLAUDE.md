# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research repository for **Universal Sequence Preconditioning** applied to time series forecasting, built on top of the **Uni2TS** framework (Salesforce's universal time series forecasting library). The research implements polynomial preconditioning (Chebyshev and Legendre) to improve time series model training by transforming the input space.

**Key Research Papers**:
1. Marsden, A., & Hazan, E. (2025). Universal Sequence Preconditioning. arXiv:2502.06545.
2. Woo, G., et al. (2024). Unified Training of Universal Time Series Forecasting Transformers. ICML 2024.

**Base Framework**: Uni2TS - PyTorch/Lightning-based library for pre-training, fine-tuning, and evaluation of Universal Time Series Transformers (Moirai models).

## Research Papers Summary

### Paper 1: Universal Sequence Preconditioning (Marsden & Hazan, 2025)

**Core Contribution**: Introduces a universal preconditioning method for sequential prediction that convolves input sequences with coefficients from orthogonal polynomials (Chebyshev or Legendre).

**Key Concepts**:
- **Preconditioning Formula**: `ỹₜ = yₜ - Σᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ` where coefficients `cᵢ` come from n-th degree monic Chebyshev/Legendre polynomial
- **Intuition**: For linear dynamical systems (LDS), preconditioning applies a polynomial to the hidden transition matrix, potentially shrinking the spectral domain and making the system "easier to learn"
- **Universal Property**: Coefficients are fixed (not learned) and work across different systems without knowing the specific system parameters

**Theoretical Results**:
- First dimension-independent sublinear regret bounds for marginally stable, asymmetric linear dynamical systems
- Two main algorithms: (1) USP + Regression achieving O(T^(-2/13)) regret, (2) USP + Spectral Filtering achieving O(T^(-3/13)) regret
- Works for systems with eigenvalues having imaginary parts bounded by O(1/log T)
- Recommended polynomial degree: 2-10 (coefficients grow exponentially as 2^(0.3n), limiting practical degree)

**Empirical Validation**:
- Tested on synthetic LDS, nonlinear dynamical systems, deep RNNs, and real ETTh1 dataset
- Consistent improvements across regression, spectral filtering, and neural network predictors
- Chebyshev and Legendre polynomials yield nearly identical performance
- Best performance typically at degrees 5-10 before coefficient growth degrades results

### Paper 2: Unified Training of Universal Time Series Forecasting Transformers (Woo et al., 2024)

**Core Contribution**: Introduces MOIRAI (Masked EncOder-based UnIveRsAl TIme Series Forecasting Transformer), a foundation model for universal time series forecasting trained on the large-scale LOTSA dataset.

**Key Architectural Innovations**:
1. **Multi Patch Size Projection**: Different patch sizes (8, 16, 32, 64, 128) for different frequencies
   - Larger patches for high-frequency data (efficient processing)
   - Smaller patches for low-frequency data (preserve temporal detail)

2. **Any-variate Attention**: Handles arbitrary number of variates by "flattening" multivariate series
   - Uses Rotary Position Embeddings (RoPE) for time encoding
   - Learned binary attention biases for variate disambiguation
   - Ensures permutation equivariance w.r.t. variate ordering

3. **Mixture Distribution Output**: Flexible probabilistic forecasting
   - Components: Student's t, Negative Binomial, Log-Normal, Low-variance Normal
   - Adapts to different data characteristics (symmetric, count, skewed, deterministic)

**LOTSA Dataset**: 27+ billion observations across 9 domains, 105 datasets
- Domains: Energy, Transport, Climate, CloudOps, Web, Sales, Nature, Economics/Finance, Healthcare
- Frequencies: Yearly to second-level (8 frequency categories)
- 59% energy domain, 72% hourly frequency, 25% minute-level

**Training Methodology**:
- Optimize mixture distribution log-likelihood
- Random context/prediction length sampling during training
- Sequence packing for efficiency (reduces padding from 61% to 0.38%)
- Three model sizes: Small (14M params), Base (91M), Large (311M)

**Empirical Results**:
- Strong zero-shot performance competitive with full-shot baselines
- Outperforms all baselines on Monash benchmark (in-distribution)
- Competitive on LSF benchmark and probabilistic forecasting tasks (out-of-distribution)

## Research Experiment Design

### Primary Research Question
**Can Universal Sequence Preconditioning improve the Moirai-Small model's forecasting performance?**

### Experimental Approach

This repository implements a controlled comparison experiment:

**1. Baseline Training** (no preconditioning):
- Pre-train Moirai-Small from scratch on LOTSA dataset
- Standard architecture without any input transformation
- Script: `pretraining/pretrain_moirai.slurm`

**2. Preconditioned Training** (with polynomial preconditioning):
- Pre-train Moirai-Small with Chebyshev/Legendre preconditioning
- Test multiple polynomial degrees (typically 2, 3, 5, 7, 10)
- Script: `pretraining/pretrain_moirai_precond.slurm`
- Variants: Test both Chebyshev and Legendre polynomials

**3. Comprehensive Evaluation**:
- Evaluate both baseline and preconditioned models on Monash benchmark datasets
- Use standard evaluation (with reversal) for fair comparison in original space
- Scripts: `eval/eval_comprehensive.slurm`
- Metrics: MAE, MSE, CRPS, MSIS across 29+ datasets

### Key Experimental Variables

**Independent Variables**:
- Preconditioning type: None (baseline), Chebyshev, Legendre
- Polynomial degree: 2, 3, 5, 7, 10

**Dependent Variables**:
- Forecasting accuracy: MAE, MSE, CRPS, MSIS
- Cross-domain generalization: Performance across different dataset domains
- Different prediction horizons: 96, 192, 336, 720 timesteps

**Controlled Variables**:
- Model architecture: Moirai-Small (384 dim, 6 layers, 14M params)
- Training data: LOTSA (27B observations)
- Context length: 1000-2000 timesteps
- Patch sizes: Adaptive (8, 16, 32, 64, 128 based on frequency)

### Expected Outcomes

Based on theoretical results from the Universal Sequence Preconditioning paper:
- Preconditioning should reduce prediction error by transforming the learning space
- Optimal polynomial degree likely in range 5-10 (balancing shrinkage vs coefficient growth)
- Improvements should be consistent across diverse datasets if time series exhibit LDS-like properties
- May see diminishing returns or degradation at very high degrees (>10) due to exponential coefficient growth

## Environment Setup

### HPC Cluster (Princeton PLI)

This repository runs on Princeton's PLI cluster using SLURM.

**Module loads** (required before every session):
```bash
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
```

**Virtual environment**:
```bash
source uni2ts/venv/bin/activate
```

**Request GPU** (interactive):
```bash
salloc --nodes=1 --ntasks=1 --mem=128G --time=03:01:00 --gres=gpu:1 --partition=pli --account=eladgroup
```

**Environment variables** (configured in `uni2ts/.env`):
- `LOTSA_V1_PATH`: Path to LOTSA (Large-scale Open Time Series Archive) dataset
- `LSF_PATH`: Path to Long Sequence Forecasting benchmark datasets

## Core Architecture

### Directory Structure

```
.
├── uni2ts/                          # Main framework code
│   ├── cli/                         # Command-line interface scripts
│   │   ├── train.py                 # Training entry point (Hydra-based)
│   │   ├── eval.py                  # Standard evaluation
│   │   ├── eval_precond_space.py    # Evaluation in transformed space
│   │   └── eval_precond_hybrid.py   # Hybrid base+precond evaluation
│   ├── src/uni2ts/
│   │   ├── model/                   # Model implementations
│   │   │   ├── moirai/              # Moirai model (Salesforce foundation model)
│   │   │   ├── moirai_moe/          # Moirai Mixture-of-Experts variant
│   │   │   └── moirai2/             # Moirai 2.0
│   │   ├── transform/               # Data transformations
│   │   │   └── precondition.py      # Polynomial preconditioning implementation
│   │   ├── data/                    # Data loading and processing
│   │   ├── loss/                    # Loss functions
│   │   └── distribution/            # Output distributions
│   └── cli/conf/                    # Hydra configuration files
│       ├── pretrain/                # Pre-training configs
│       ├── finetune/                # Fine-tuning configs
│       └── eval/                    # Evaluation configs
├── *.slurm                          # SLURM batch job scripts
├── eval_confs/                      # Evaluation dataset configurations
├── logs/                            # SLURM job outputs
└── Time-Series-Library/             # External benchmark datasets
```

### Key Components

**Preconditioning Transform** (`uni2ts/src/uni2ts/transform/precondition.py`):
- Implements Universal Sequence Preconditioning using polynomial convolutions
- Formula: `ỹₜ = yₜ - Σᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ` where `cᵢ` are polynomial coefficients
- Supports Chebyshev and Legendre polynomials
- Respects series boundaries (no cross-contamination between time series)
- Recommended degree: 2-10 (paper suggests ≤10 for numerical stability)

**Model Architecture**:
- Based on Transformer architecture with patching
- Patch sizes: 8, 16, 32, 64, 128 (adaptive)
- Context length: typically 1000-2000 timesteps
- Output distributions: Mixture of StudentT, Normal, NegativeBinomial, LogNormal
- Model variants: small (384 dim, 6 layers), base, large

**Training Framework**:
- PyTorch Lightning for training loop
- Hydra for configuration management
- Distributed training support via PyTorch DDP
- Supports both pre-training (from scratch) and fine-tuning (from checkpoints)

## Common Commands

### Core Experimental Scripts

**Submit baseline pre-training** (no preconditioning):
```bash
cd /scratch/gpfs/EHAZAN/jh1161
sbatch pretraining/pretrain_moirai.slurm
```

**Submit preconditioned pre-training** (Chebyshev or Legendre):
```bash
cd /scratch/gpfs/EHAZAN/jh1161
# Default: Chebyshev degree 5
sbatch pretraining/pretrain_moirai_precond.slurm

# Custom degree/type
sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=7 pretraining/pretrain_moirai_precond.slurm
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=5 pretraining/pretrain_moirai_precond.slurm
```

**Submit comprehensive evaluation on Monash datasets**:
```bash
cd /scratch/gpfs/EHAZAN/jh1161
# Evaluate baseline model
sbatch --export=MODEL_PATH=/path/to/baseline_checkpoint.ckpt eval/eval_comprehensive.slurm

# Evaluate preconditioned model
sbatch --export=MODEL_PATH=/path/to/precond_checkpoint.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 eval/eval_precond_comprehensive.slurm
```

### Training (Interactive/Direct)

**Baseline pre-training** (no preconditioning):
```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -m cli.train \
  -cp conf/pretrain \
  run_name=baseline_run \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  seed=0
```

**Pre-training with preconditioning**:
```bash
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_chebyshev_d5 \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  model.enable_preconditioning=true \
  model.precondition_type=chebyshev \
  model.precondition_degree=5 \
  seed=0
```

**Fine-tuning** on a specific dataset:
```bash
python -m cli.train \
  -cp conf/finetune \
  exp_name=lsf_finetune \
  run_name=etth1_run \
  model=moirai_1.0_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  model.prediction_length=96 \
  data=etth1 \
  val_data=etth1
```

### Evaluation

**Standard evaluation** (with preconditioning reversal):
```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -m cli.eval \
  run_name=eval_standard \
  model=moirai_1.0_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96
```

**Preconditioned space evaluation** (no reversal):
```bash
python -m cli.eval_precond_space \
  model=moirai_precond_ckpt_no_reverse \
  model.checkpoint_path=/path/to/checkpoint.ckpt \
  model.patch_size=32 \
  model.context_length=1000 \
  model.precondition_type=chebyshev \
  model.precondition_degree=5 \
  data=monash_cached \
  data.dataset_name=m1_monthly
```

**Hybrid evaluation** (base + preconditioned):
```bash
python -m cli.eval_precond_hybrid \
  run_name=hybrid_eval \
  base_model=moirai_1.1_R_small \
  precond_model.checkpoint_path=/path/to/precond.ckpt \
  precond_model.precondition_type=chebyshev \
  precond_model.precondition_degree=5 \
  precond_model.reverse_output=false \
  data=monash_cached \
  data.dataset_name=m1_monthly
```

### SLURM Batch Jobs

**Submit pre-training job**:
```bash
cd /scratch/gpfs/EHAZAN/jh1161
sbatch pretrain_moirai_precond.slurm
```

**With custom parameters**:
```bash
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=7 pretrain_moirai_precond.slurm
```

**Submit comprehensive evaluation**:
```bash
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt eval_precond_comprehensive.slurm
```

**Monitor jobs**:
```bash
squeue -u $USER                      # Check queue
tail -f logs/pretrain_*.out          # Monitor log
scancel JOBID                        # Cancel job
```

### Testing

**Run single dataset test**:
```bash
cd /scratch/gpfs/EHAZAN/jh1161
bash test_config_loading.sh
```

**Test preconditioning pipeline**:
```bash
python test_full_precond_pipeline.py
```

**Debug specific dataset**:
```bash
python debug_rideshare.py
```

## Evaluation Approaches

This repository implements **three distinct evaluation methodologies**:

### 1. Standard Evaluation (`cli/eval.py`)
- Predictions in original space (with automatic reversal if model was trained with preconditioning)
- Compare predictions vs original ground truth
- Use for: end-user metrics, baseline comparisons

### 2. Preconditioned Space Evaluation (`cli/eval_precond_space.py`)
- Predictions in transformed/preconditioned space (no reversal)
- Compare transformed predictions vs transformed ground truth
- Use for: understanding model performance in training space, comparing preconditioned models fairly
- Set `model.reverse_output=false` when loading model

### 3. Hybrid Evaluation (`cli/eval_precond_hybrid.py`)
- Combines base model + preconditioned model predictions
- Uses base model's predictions as context when reversing preconditioned model's output
- Formula: `y_hybrid[t] = ỹ_precond[t] + Σ cᵢ · y_base[t-i]`
- Use for: residual modeling, model ensembling, transfer learning

**Comprehensive evaluation**: All three approaches have corresponding `*_comprehensive.slurm` scripts that evaluate across 29+ benchmark datasets (M1, M3, M4, Tourism, NN5, Traffic, etc.) and automatically aggregate metrics into CSV files.

## Dataset Configuration

Evaluation datasets are configured in `eval_confs/forecast_datasets.xlsx` (read by `read_datasets_config.py`). This specifies:
- Dataset display names
- Internal dataset identifiers for Monash archive
- Prediction lengths for each dataset
- Evaluation order

The datasets are cached locally in `/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1` to enable **offline mode** (no HuggingFace Hub access required during jobs).

## Important Implementation Details

### Preconditioning

**Series Boundary Safety**: The preconditioning implementation correctly handles multiple time series in a batch:
- Each series is processed independently
- No coefficients are computed across series boundaries
- Verified in `SERIES_BOUNDARY_VERIFICATION.md`

**Reversal Modes**:
- During training: preconditioning applied, no reversal
- During standard evaluation: preconditioning applied, then reversed before computing metrics
- During preconditioned space evaluation: preconditioning applied, no reversal
- Control with `model.reverse_output=true/false` or `model.enable_preconditioning=false`

### Configuration Management (Hydra)

All scripts use Hydra for configuration:
- Configurations in `uni2ts/cli/conf/`
- Override with command-line args: `key=value`
- Nested configs: `model.patch_size=32`
- Multiple configs: `-cp conf/pretrain` (change config path)

**Key config parameters**:
- `model`: Model architecture (moirai_small, moirai_base, moirai_large)
- `data`: Training data source (lotsa_v1_unweighted, monash_cached, etc.)
- `model.enable_preconditioning`: Enable/disable preconditioning
- `model.precondition_type`: chebyshev or legendre
- `model.precondition_degree`: 2-10 (recommended ≤10)
- `model.patch_size`: 8, 16, 32, 64, or 128
- `model.context_length`: Input context window size
- `data.prediction_length`: Forecast horizon

### Output Locations

**Training outputs**: `uni2ts/outputs/<run_name>/`
- Checkpoints: `*.ckpt` files
- Logs: TensorBoard logs
- Configs: Hydra config snapshots

**Evaluation outputs**: `eval_*_results_*/`
- Metrics CSV: aggregated metrics across datasets
- Individual outputs: `<dataset>_output.txt` per dataset

**SLURM logs**: `logs/`
- Standard output: `*_<jobid>.out`
- Error output: `*_<jobid>.err`

## Known Issues and Workarounds

**NaN handling**: Some datasets (e.g., Rideshare) contain NaN values in ground truth. Evaluation scripts filter these out and report status as `all_nan`, `partial_success`, or `failed` in results CSV.

**Memory issues**: If OOM errors occur during evaluation, reduce `BATCH_SIZE` environment variable when submitting SLURM jobs.

**Offline mode**: All comprehensive evaluation scripts set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prevent network access. Ensure datasets are cached locally first.

**Checkpoint loading**: Training checkpoints use Lightning format. For evaluation, specify `model.checkpoint_path=/path/to/file.ckpt`.

## Parameter Sweeps

Use `submit_precond_sweep.sh` to run systematic experiments across multiple preconditioning configurations:
```bash
cd /scratch/gpfs/EHAZAN/jh1161
bash submit_precond_sweep.sh
```

This submits 7 jobs:
- 1 baseline (no preconditioning)
- 5 Chebyshev degrees (2, 3, 5, 7, 10)
- 1 Legendre comparison (degree 5)

Results can be compared across `uni2ts/outputs/precond_*_<timestamp>/` directories.

## Documentation Index

Key reference documents in repository:
- `HYBRID_EVALUATION_README.md`: Hybrid evaluation methodology
- `EVAL_PRECOND_README.md`: Preconditioned space evaluation
- `QUICKSTART_PRECONDITIONING.md`: Quick reference for common preconditioning tasks
- `README_SCRIPTS.md`: Overview of SLURM scripts
- `SLURM_PRECONDITIONING_GUIDE.md`: Detailed SLURM usage guide
- `uni2ts/README.md`: Upstream Uni2TS framework documentation
- `PRECONDITIONING_QUICK_REFERENCE.md`: Mathematical formulation reference

## Development Workflow

1. **Experiment setup**: Modify or create SLURM scripts for your configuration
2. **Submit job**: `sbatch <script>.slurm` with appropriate environment variables
3. **Monitor**: Use `squeue -u $USER` and `tail -f logs/...` to track progress
4. **Analyze results**: Check `uni2ts/outputs/` for training results or `eval_*_results_*/` for evaluation metrics
5. **Compare**: Use plotting scripts like `plot_training_comparison.py` or analyze CSV metrics
6. **Iterate**: Adjust hyperparameters and resubmit

## Testing Before Large Runs

Always test on a single dataset before submitting comprehensive evaluation jobs:
```bash
# Test single dataset interactively
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -m cli.eval_precond_space \
  model=moirai_precond_ckpt_no_reverse \
  model.checkpoint_path=/path/to/checkpoint.ckpt \
  model.precondition_type=chebyshev \
  model.precondition_degree=5 \
  data=monash_cached \
  data.dataset_name=m1_monthly
```

If successful, then submit the comprehensive SLURM job.
