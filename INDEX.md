# Complete Documentation Index

**Quick navigation to all scripts and documentation**

---

## 🚀 Quick Start

**New to this?** Start here:

1. **Training**: Read `QUICKSTART_PRECONDITIONING.md`
2. **Evaluation**: Read `QUICKSTART_EVALUATION.md`
3. **Complete Flow**: Read `COMPLETE_WORKFLOW.md`

---

## 📂 File Listing

### Training Scripts (4 files)

| File | Purpose | Time |
|------|---------|------|
| `pretrain_moirai.slurm` | Baseline (no precond) | 48h |
| `pretrain_moirai_precond_default.slurm` | Default precond ⭐ | 48h |
| `pretrain_moirai_precond.slurm` | Custom precond | 48h |
| `submit_precond_sweep.sh` | Full sweep (7 jobs) | 336h |

### Evaluation Scripts (5 files)

| File | Purpose | Time |
|------|---------|------|
| `eval_moirai_checkpoint.slurm` | Single dataset | 1-2h |
| `eval_moirai_by_frequency.slurm` | One frequency | 4-8h |
| `eval_moirai_monash_frequencies.slurm` | All freq (sequential) | 12-24h |
| `submit_eval_all_frequencies.sh` | All freq (parallel) ⭐ | 8h |
| `find_checkpoint.sh` | Find checkpoints | instant |

### Documentation (8 files)

| File | What It Covers |
|------|----------------|
| `QUICKSTART_PRECONDITIONING.md` | Training quick ref |
| `SLURM_PRECONDITIONING_GUIDE.md` | Training full guide |
| `QUICKSTART_EVALUATION.md` | Eval quick ref |
| `EVALUATION_GUIDE.md` | Eval full guide |
| `COMPLETE_WORKFLOW.md` | End-to-end workflow |
| `README_SCRIPTS.md` | Scripts overview |
| `SERIES_BOUNDARY_VERIFICATION.md` | Implementation details |
| `INDEX.md` | This file |

---

## 🎯 Use Cases

### I want to... → Read this

| Task | Document | Script |
|------|----------|--------|
| Train with default preconditioning | `QUICKSTART_PRECONDITIONING.md` | `pretrain_moirai_precond_default.slurm` |
| Train with custom parameters | `SLURM_PRECONDITIONING_GUIDE.md` | `pretrain_moirai_precond.slurm` |
| Run full parameter sweep | `SLURM_PRECONDITIONING_GUIDE.md` | `submit_precond_sweep.sh` |
| Find my trained checkpoint | `QUICKSTART_EVALUATION.md` | `find_checkpoint.sh` |
| Evaluate on all frequencies | `QUICKSTART_EVALUATION.md` | `submit_eval_all_frequencies.sh` |
| Evaluate on specific datasets | `EVALUATION_GUIDE.md` | `eval_moirai_by_frequency.slurm` |
| Understand full workflow | `COMPLETE_WORKFLOW.md` | - |
| Learn about implementation | `SERIES_BOUNDARY_VERIFICATION.md` | - |

---

## 📖 Reading Order

### For Beginners

1. `COMPLETE_WORKFLOW.md` - Get the big picture
2. `QUICKSTART_PRECONDITIONING.md` - Start training
3. `QUICKSTART_EVALUATION.md` - Evaluate results

### For Power Users

1. `SLURM_PRECONDITIONING_GUIDE.md` - All training options
2. `EVALUATION_GUIDE.md` - All evaluation options
3. `SERIES_BOUNDARY_VERIFICATION.md` - Implementation details

---

## 🔧 Common Commands

### Training
```bash
# Default preconditioning
sbatch pretrain_moirai_precond_default.slurm

# Custom parameters
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=10 pretrain_moirai_precond.slurm

# Full sweep
bash submit_precond_sweep.sh
```

### Evaluation
```bash
# Find checkpoint
bash find_checkpoint.sh

# Evaluate all
bash submit_eval_all_frequencies.sh /path/to/checkpoint.ckpt

# Evaluate yearly only
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt,FREQUENCY=yearly eval_moirai_by_frequency.slurm
```

### Monitoring
```bash
# Check jobs
squeue -u $USER

# View logs
tail -f logs/*.out

# Check results
ls -lh uni2ts/outputs/
```

---

## 📊 File Organization

```
/scratch/gpfs/EHAZAN/jh1161/
│
├── Training Scripts
│   ├── pretrain_moirai.slurm
│   ├── pretrain_moirai_precond_default.slurm
│   ├── pretrain_moirai_precond.slurm
│   └── submit_precond_sweep.sh
│
├── Evaluation Scripts
│   ├── eval_moirai_checkpoint.slurm
│   ├── eval_moirai_by_frequency.slurm
│   ├── eval_moirai_monash_frequencies.slurm
│   ├── submit_eval_all_frequencies.sh
│   └── find_checkpoint.sh
│
├── Documentation
│   ├── Quick Start
│   │   ├── QUICKSTART_PRECONDITIONING.md
│   │   ├── QUICKSTART_EVALUATION.md
│   │   └── COMPLETE_WORKFLOW.md
│   │
│   ├── Full Guides
│   │   ├── SLURM_PRECONDITIONING_GUIDE.md
│   │   ├── EVALUATION_GUIDE.md
│   │   └── README_SCRIPTS.md
│   │
│   └── Technical
│       ├── SERIES_BOUNDARY_VERIFICATION.md
│       └── INDEX.md (this file)
│
├── Logs
│   ├── logs/pretrain_*.{out,err}
│   └── logs/eval_*.{out,err}
│
└── Outputs
    └── uni2ts/outputs/
        ├── Training runs
        │   ├── baseline_*/checkpoints/
        │   └── precond_*/checkpoints/
        │
        └── Evaluation results
            └── eval_*/metrics.csv
```

---

## 🎓 Learning Path

### Day 1: Setup & First Run
1. Read `COMPLETE_WORKFLOW.md`
2. Run: `sbatch pretrain_moirai_precond_default.slurm`
3. Monitor with: `tail -f logs/pretrain_*.out`

### Day 2-3: Wait for Training
- Check job status periodically
- Review documentation
- Plan evaluation strategy

### Day 4: Evaluation
1. Run: `bash find_checkpoint.sh`
2. Run: `bash submit_eval_all_frequencies.sh /path/to/ckpt`
3. Monitor with: `tail -f logs/eval_*.out`

### Day 5: Analysis
1. View results: `ls uni2ts/outputs/eval_*/`
2. Compare metrics
3. Plan next experiments

---

## 📝 Cheat Sheet

### Fastest Path to Results

```bash
# 1. Train (48h)
sbatch pretrain_moirai_precond_default.slurm

# 2. Find checkpoint
bash find_checkpoint.sh precond_default

# 3. Evaluate (8h)
bash submit_eval_all_frequencies.sh outputs/precond_default_*/checkpoints/last.ckpt

# 4. Check results
ls uni2ts/outputs/eval_*/metrics.csv
```

**Total time**: ~56 hours from start to results

---

## 🆘 Getting Help

### Something not working?

1. **Check the appropriate guide**:
   - Training issue? → `SLURM_PRECONDITIONING_GUIDE.md`
   - Evaluation issue? → `EVALUATION_GUIDE.md`

2. **Check logs**:
   ```bash
   tail -100 logs/*.err
   ```

3. **Verify setup**:
   ```bash
   bash find_checkpoint.sh
   squeue -u $USER
   ```

---

## 🔗 Related Files

### In uni2ts/ directory

Additional documentation in the project directory:

- `uni2ts/PRECONDITIONING_USAGE.md` - API usage examples
- `uni2ts/PRECONDITIONING_IMPLEMENTATION_SUMMARY.md` - Implementation overview
- `uni2ts/SERIES_BOUNDARY_VERIFICATION.md` - Safety verification

---

## ✅ Status Overview

All files created and verified:

- ✅ 4 Training scripts
- ✅ 5 Evaluation scripts
- ✅ 8 Documentation files
- ✅ All scripts are executable
- ✅ All paths verified
- ✅ Ready to use!

---

## 🚦 Getting Started RIGHT NOW

```bash
cd /scratch/gpfs/EHAZAN/jh1161

# Read this first
cat QUICKSTART_PRECONDITIONING.md

# Then run your first training
sbatch pretrain_moirai_precond_default.slurm

# Check it's running
squeue -u $USER
```

That's it! Come back in ~48 hours for evaluation. 🎉

---

**Quick Links**:
- Training: `QUICKSTART_PRECONDITIONING.md`
- Evaluation: `QUICKSTART_EVALUATION.md`
- Full Flow: `COMPLETE_WORKFLOW.md`

**Last Updated**: 2025-11-01
