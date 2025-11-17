# Evaluation System Summary

## ✅ Complete Evaluation System for Baseline vs Preconditioned Model Comparison

All scripts are configured, tested, and ready to use.

---

## 📋 Files Created/Updated

### Core Scripts
1. ✅ **`read_datasets_config.py`** - Dataset configuration parser (29 datasets validated)
2. ✅ **`eval_comprehensive.slurm`** - Standard evaluation (original space)
3. ✅ **`eval_precond_comprehensive.slurm`** - Preconditioned model evaluation (transformed space)
4. ✅ **`eval_baseline_in_precond_space.slurm`** - NEW: Baseline in preconditioned space
5. ✅ **`uni2ts/cli/eval_baseline_in_precond_space.py`** - NEW: Python implementation

### Helper Tools
6. ✅ **`compare_models.sh`** - Submit all 4 evaluation jobs at once
7. ✅ **`compare_csv_results.py`** - Compare CSV outputs and compute improvements

### Documentation
8. ✅ **`README.md`** - Main documentation
9. ✅ **`EVALUATION_COMPARISON_GUIDE.md`** - Detailed comparison methodology
10. ✅ **`DATASET_MAPPING.md`** - Dataset name mappings explained

---

## 🎯 Key Features Implemented

### 1. Frequency-Based Patch Sizing (Per Moirai Paper)
```
Yearly/Quarterly (Y, Q):           patch_size = 8
Monthly/Weekly/Daily/Hourly:       patch_size = 32
Minute/Second-level (30T, 5T):     patch_size = 64
```

**Datasets affected**:
- Tourism Quarterly → patch_size=8
- Aus. Elec. Demand (30T) → patch_size=64
- All others → patch_size=32

### 2. Correct Dataset Path Mapping
All 29 datasets validated against cached data in:
`/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1/`

**Special mappings** (15 datasets):
- `Aus. Elec. Demand` → `australian_electricity_demand`
- `M3 Monthly` → `monash_m3_monthly`
- `Bitcoin` → `bitcoin_with_missing`
- etc. (see DATASET_MAPPING.md)

### 3. Organized Output Structure
All results saved to:
```
eval/outputs/
├── eval_results_*_*/                      # Standard evaluation
├── eval_precond_results_*_*/              # Precond model (transformed space)
└── eval_baseline_precond_space_*_*/       # Baseline (transformed space)
```

### 4. Robust CSV Parsing
- Extracts metrics from text output
- Handles partial failures gracefully
- Consistent format across all three evaluation modes

---

## 🚀 Quick Start Guide

### Option 1: Full Comparison (Automated)

Submit all 4 evaluation jobs with one command:

```bash
cd /scratch/gpfs/EHAZAN/jh1161

bash eval/compare_models.sh \
  /path/to/baseline.ckpt \
  /path/to/precond.ckpt \
  chebyshev \
  5
```

This submits:
1. Baseline → Original space
2. Preconditioned → Original space (reversed)
3. Baseline → Preconditioned space (post-hoc)
4. Preconditioned → Preconditioned space (native)

### Option 2: Individual Evaluations

**Standard evaluation** (original space):
```bash
sbatch --export=MODEL_PATH=/path/to/model.ckpt eval/eval_comprehensive.slurm
```

**Preconditioned space** (preconditioned model):
```bash
sbatch --export=MODEL_PATH=/path/to/precond.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 \
  eval/eval_precond_comprehensive.slurm
```

**Preconditioned space** (baseline model):
```bash
sbatch --export=MODEL_PATH=/path/to/baseline.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 \
  eval/eval_baseline_in_precond_space.slurm
```

### Compare Results

```bash
python eval/compare_csv_results.py \
  --baseline-orig "eval/outputs/eval_results_*baseline*/evaluation_metrics.csv" \
  --precond-orig "eval/outputs/eval_results_*precond*/evaluation_metrics.csv" \
  --baseline-precond "eval/outputs/eval_baseline_precond_space_*/evaluation_metrics_baseline_in_precond_space.csv" \
  --precond-precond "eval/outputs/eval_precond_results_*/evaluation_metrics_precond_space.csv" \
  --output comparison_summary.csv
```

---

## 📊 What Each Comparison Tells You

### Comparison 1: Original Space (Jobs 1 vs 2)
**Question**: Which model is better for end-users?

- Compare: `eval_results_baseline_*/evaluation_metrics.csv` vs `eval_results_precond_*/evaluation_metrics.csv`
- Metric: `MAE_median`, `MSE_mean`, etc.
- **If preconditioned wins** → Preconditioning improves practical forecasting

### Comparison 2: Preconditioned Space (Jobs 3 vs 4)
**Question**: Did training on preconditioned data help learn the transformation better?

- Compare: `eval_baseline_precond_space_*/...csv` vs `eval_precond_results_*/...csv`
- Both evaluated in transformed space
- **If preconditioned wins** → Training on transformed data is beneficial, not just the transformation itself

---

## 🔧 Troubleshooting

### Dataset Not Found Error
If you see: `FileNotFoundError: Cached dataset not found at ...`

**Solution**: Check dataset mapping in `read_datasets_config.py`
```bash
# Validate all datasets
cd /scratch/gpfs/EHAZAN/jh1161/eval
python3 read_datasets_config.py | python3 -c "
import subprocess, json, sys
data = json.loads(sys.stdin.read())
cache = set(subprocess.run(['ls', '/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1'],
            capture_output=True, text=True).stdout.strip().split('\n'))
missing = [d for d in data if d['dataset_name'] not in cache]
if missing:
    print('Missing datasets:')
    for d in missing:
        print(f'  {d[\"display_name\"]} -> {d[\"dataset_name\"]}')
else:
    print('✓ All datasets valid!')
"
```

### Job Fails on Specific Dataset
- Check individual output file: `eval/outputs/*/DatasetName_output.txt`
- Dataset may have NaN values in ground truth (e.g., Rideshare)
- Script will continue to next dataset automatically

---

## 📈 Expected Output

### Successful Run Shows:
```
[1/29] Evaluating: M1 Monthly (dataset: m1_monthly, freq: M, pred_len: 18, patch: 32)
  ✓ M1 Monthly completed successfully

[8/29] Evaluating: Tourism Quarterly (dataset: tourism_quarterly, freq: Q, pred_len: 8, patch: 8)
  ✓ Tourism Quarterly completed successfully

[11/29] Evaluating: Aus. Elec. Demand (dataset: australian_electricity_demand, freq: 30T, pred_len: 336, patch: 64)
  ✓ Aus. Elec. Demand completed successfully

...

Summary:
  Total datasets: 29
  Successful: 25
  Failed: 4
```

### CSV Output:
```
dataset,MSE_mean,MSE_median,MAE_median,...,status
M1 Monthly,420270210.345,820324714.966,2451.959,...,success
Tourism Quarterly,8167097138.061,8272179311.363,15863.879,...,success
Aus. Elec. Demand,12345.67,23456.78,123.45,...,success
...
```

---

## ✅ Validation Checklist

- [x] All 29 datasets map to valid cache paths
- [x] Frequency-based patch sizing implemented (Q=8, M/W/D/H=32, *T/*S=64)
- [x] Outputs save to `eval/outputs/` subdirectory
- [x] CSV parsing extracts metrics correctly
- [x] Three evaluation modes working (original, precond native, baseline→precond)
- [x] Helper scripts for automation and comparison
- [x] Complete documentation provided

---

## 🎓 Research Workflow

```
┌─────────────┐
│  Pre-train  │
│   Models    │
└──────┬──────┘
       │
       ├─→ Baseline (no precond)  → baseline.ckpt
       └─→ Preconditioned (deg 5) → precond.ckpt
              │
              ▼
       ┌─────────────┐
       │  Evaluate   │
       │ (4 modes)   │
       └──────┬──────┘
              │
              ├─→ Baseline in original space      → CSV 1
              ├─→ Precond in original space       → CSV 2
              ├─→ Baseline in precond space       → CSV 3
              └─→ Precond in precond space        → CSV 4
                     │
                     ▼
              ┌─────────────┐
              │   Compare   │
              │   Results   │
              └──────┬──────┘
                     │
                     └─→ Improvement analysis
                         - Original space: CSV1 vs CSV2
                         - Precond space:  CSV3 vs CSV4
```

**Ready to run experiments!** 🚀
