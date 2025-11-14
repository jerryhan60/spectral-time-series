# SLURM Scripts Summary

**Created**: 2025-11-01
**Location**: `/scratch/gpfs/EHAZAN/jh1161/`

---

## What Was Created

### SLURM Scripts (3 files)

1. **`pretrain_moirai_precond.slurm`**
   - Flexible preconditioning with customizable parameters
   - Pass parameters via `--export=PRECOND_TYPE=X,PRECOND_DEGREE=Y`
   - Default: Chebyshev degree 5

2. **`pretrain_moirai_precond_default.slurm`**
   - Uses preconfigured model `moirai_small_precond`
   - Simplest option for getting started
   - Default: Chebyshev degree 5 (from config)

3. **`pretrain_moirai.slurm`** *(existing)*
   - Baseline without preconditioning
   - For comparison purposes

### Helper Scripts (1 file)

4. **`submit_precond_sweep.sh`** *(executable)*
   - Submits complete parameter sweep
   - 7 jobs total: baseline + 6 preconditioning variants
   - Chebyshev degrees: 2, 3, 5, 7, 10
   - Legendre degree: 5

### Documentation (3 files)

5. **`SLURM_PRECONDITIONING_GUIDE.md`**
   - Comprehensive guide (10KB)
   - Covers all usage scenarios
   - Troubleshooting section
   - Best practices

6. **`QUICKSTART_PRECONDITIONING.md`**
   - Quick reference for common tasks
   - 3 main usage patterns
   - Essential commands only

7. **`README_SCRIPTS.md`** *(this file)*
   - Overview of all created files
   - Quick navigation guide

---

## Quick Start

### Run Single Experiment
```bash
cd /scratch/gpfs/EHAZAN/jh1161
sbatch pretrain_moirai_precond_default.slurm
```

### Run Full Sweep
```bash
cd /scratch/gpfs/EHAZAN/jh1161
bash submit_precond_sweep.sh
```

### Check Status
```bash
squeue -u $USER
tail -f logs/pretrain_precond_*.out
```

---

## File Tree

```
/scratch/gpfs/EHAZAN/jh1161/
├── pretrain_moirai.slurm                       # Existing baseline
├── pretrain_moirai_precond.slurm              # NEW: Custom parameters
├── pretrain_moirai_precond_default.slurm      # NEW: Default config
├── pretrain_moirai_multi_gpu.slurm            # Existing multi-GPU
├── submit_precond_sweep.sh                     # NEW: Sweep helper
├── SLURM_PRECONDITIONING_GUIDE.md             # NEW: Full guide
├── QUICKSTART_PRECONDITIONING.md              # NEW: Quick reference
├── README_SCRIPTS.md                           # NEW: This file
├── logs/                                       # Job logs
│   ├── pretrain_*.out
│   └── pretrain_*.err
└── uni2ts/
    ├── outputs/                                # Results
    ├── PRECONDITIONING_USAGE.md               # Usage examples
    ├── SERIES_BOUNDARY_VERIFICATION.md        # Implementation details
    └── cli/conf/pretrain/model/
        └── moirai_small_precond.yaml          # Precond config
```

---

## Navigation Guide

| Task | Document |
|------|----------|
| **Quick start** | `QUICKSTART_PRECONDITIONING.md` |
| **Full documentation** | `SLURM_PRECONDITIONING_GUIDE.md` |
| **Implementation details** | `uni2ts/SERIES_BOUNDARY_VERIFICATION.md` |
| **Usage examples** | `uni2ts/PRECONDITIONING_USAGE.md` |
| **Configuration reference** | `uni2ts/PRECONDITIONING_IMPLEMENTATION_SUMMARY.md` |

---

## Key Features

### Flexible Parameter Control
```bash
# Change polynomial type
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=5 pretrain_moirai_precond.slurm

# Change degree
sbatch --export=PRECOND_DEGREE=10 pretrain_moirai_precond.slurm

# Add suffix to run name
sbatch --export=RUN_SUFFIX=_experiment1 pretrain_moirai_precond.slurm
```

### Automatic Run Naming
All jobs get unique, descriptive names:
- `precond_chebyshev_d5_20251101_143052`
- `precond_legendre_d10_20251101_150312`
- `precond_default_20251101_120000`

### Comprehensive Logging
- Job output: `logs/pretrain_precond_JOBID.out`
- Job errors: `logs/pretrain_precond_JOBID.err`
- Training logs: `uni2ts/outputs/RUN_NAME/`

---

## Usage Patterns

### Pattern 1: Quick Test
```bash
sbatch pretrain_moirai_precond_default.slurm
```
**Use when**: First time trying preconditioning

### Pattern 2: Comparison
```bash
sbatch pretrain_moirai.slurm
sbatch pretrain_moirai_precond_default.slurm
```
**Use when**: Need baseline comparison

### Pattern 3: Parameter Sweep
```bash
bash submit_precond_sweep.sh
```
**Use when**: Exploring optimal parameters

### Pattern 4: Custom Experiment
```bash
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=7 pretrain_moirai_precond.slurm
```
**Use when**: Testing specific configuration

---

## Configuration Summary

### Default Settings (All Scripts)
- **Partition**: pli
- **Account**: eladgroup
- **Time**: 48 hours
- **Resources**: 1 GPU, 128GB RAM, 16 CPUs
- **Data**: lotsa_v1_unweighted
- **Model**: moirai_small (or moirai_small_precond)
- **Seed**: 0

### Preconditioning Defaults
- **Type**: Chebyshev
- **Degree**: 5
- **Paper recommendation**: Degree ≤ 10 for stability

---

## Experiment Workflow

### Phase 1: Initial Test
1. Run baseline: `sbatch pretrain_moirai.slurm`
2. Run default precond: `sbatch pretrain_moirai_precond_default.slurm`
3. Wait 2-4 hours, check training progress
4. Compare initial loss curves

### Phase 2: Parameter Search
1. Run sweep: `bash submit_precond_sweep.sh`
2. Monitor all jobs: `squeue -u $USER`
3. Wait for completion (~48 hours per job)
4. Compare final metrics across runs

### Phase 3: Analysis
1. Collect results from `uni2ts/outputs/`
2. Compare loss curves, metrics
3. Identify best configuration
4. Document findings

---

## Important Notes

### Series Boundary Safety
The preconditioning implementation **correctly respects series boundaries**:
- Each series processed independently
- No cross-series contamination
- Verified by tests in `SERIES_BOUNDARY_VERIFICATION.md`

### Resource Requirements
- Each job needs ~48 hours wall time
- Full sweep (7 jobs) = ~336 GPU-hours
- Can run in parallel if multiple GPUs available

### Logs Directory
Make sure it exists before submitting:
```bash
mkdir -p logs
```

---

## Support Resources

### Documentation Files
1. `QUICKSTART_PRECONDITIONING.md` - Quick commands
2. `SLURM_PRECONDITIONING_GUIDE.md` - Full guide
3. `uni2ts/PRECONDITIONING_USAGE.md` - Python API usage
4. `uni2ts/SERIES_BOUNDARY_VERIFICATION.md` - Implementation verification

### Command Reference
```bash
# Submit job
sbatch SCRIPT.slurm

# Check queue
squeue -u $USER

# View logs
tail -f logs/pretrain_*.out

# Cancel job
scancel JOBID

# Job history
sacct -u $USER
```

---

## Updates

**2025-11-01**: Initial creation
- 3 SLURM scripts
- 1 helper script
- 3 documentation files
- Verified series boundary handling

---

**Contact**: jh1161@princeton.edu
