# Quick Start: Preconditioning Experiments

**TL;DR**: Run pretraining experiments with preconditioning in 3 commands.

---

## Option 1: Single Experiment (Recommended First Try)

```bash
cd /scratch/gpfs/EHAZAN/jh1161

# Run default preconditioning (Chebyshev degree 5)
sbatch pretrain_moirai_precond_default.slurm

# Check status
squeue -u $USER

# Monitor progress
tail -f logs/pretrain_precond_default_*.out
```

---

## Option 2: Baseline vs Preconditioning Comparison

```bash
cd /scratch/gpfs/EHAZAN/jh1161

# Submit both jobs
sbatch pretrain_moirai.slurm                    # Baseline
sbatch pretrain_moirai_precond_default.slurm    # With preconditioning

# Check status
squeue -u $USER
```

---

## Option 3: Full Parameter Sweep

```bash
cd /scratch/gpfs/EHAZAN/jh1161

# Submit all experiments (7 jobs total)
bash submit_precond_sweep.sh

# Check queue
squeue -u $USER
```

This will run:
- 1 baseline (no preconditioning)
- 1 default preconditioning
- 5 Chebyshev degree sweep (2, 3, 5, 7, 10)
- 1 Legendre comparison

---

## Custom Parameters

```bash
# Legendre polynomial with degree 10
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=10 \
       pretrain_moirai_precond.slurm

# Chebyshev with degree 3
sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=3 \
       pretrain_moirai_precond.slurm
```

---

## Check Results

```bash
# View logs
ls -lth logs/

# View outputs
ls -lth uni2ts/outputs/

# Monitor specific job
tail -f logs/pretrain_precond_JOBID.out
```

---

## Cancel Jobs

```bash
# Cancel specific job
scancel JOBID

# Cancel all your jobs
scancel -u $USER
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `pretrain_moirai.slurm` | Baseline (no preconditioning) |
| `pretrain_moirai_precond_default.slurm` | Default preconditioning (easiest) |
| `pretrain_moirai_precond.slurm` | Custom parameters |
| `submit_precond_sweep.sh` | Run all experiments |
| `SLURM_PRECONDITIONING_GUIDE.md` | Full documentation |

---

## Expected Runtime

- **Per job**: ~48 hours
- **Sweep (7 jobs)**: ~336 hours sequential, ~48 hours parallel (if GPUs available)

---

## Next Steps

1. **Start with Option 1**: Run default preconditioning
2. **Wait 2-4 hours**: Check initial training progress
3. **Compare**: If working well, run full sweep (Option 3)
4. **Analyze**: Compare results across different configurations

---

## Help

**Full Guide**: See `SLURM_PRECONDITIONING_GUIDE.md` for detailed documentation

**Implementation Details**: See `SERIES_BOUNDARY_VERIFICATION.md` for how preconditioning works

**Usage Examples**: See `uni2ts/PRECONDITIONING_USAGE.md` for more usage patterns
