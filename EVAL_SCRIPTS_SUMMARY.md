# Evaluation Scripts Summary

## Overview
This document summarizes all evaluation scripts and identifies which ones to keep/archive.

---

## 📊 CURRENT EVALUATION SCRIPTS

### ✅ **KEEP - Core Scripts (Latest & Bug-Free)**

#### 1. **`eval_precond_monash.slurm`** ⭐ PRIMARY
- **Purpose**: Evaluate preconditioned models on all Monash datasets (yearly, quarterly, monthly)
- **Model Type**: Preconditioned checkpoint (with polynomial preconditioning)
- **Features**:
  - ✅ Supports preconditioning (enable_preconditioning=true)
  - ✅ Uses `moirai_precond_ckpt` config
  - ✅ Evaluates on TEST split
  - ✅ Evaluates ALL frequencies (12 datasets total)
- **Usage**:
  ```bash
  sbatch eval_precond_monash.slurm
  ```
- **Status**: ✅ Latest, includes preconditioning support

---

#### 2. **`eval_precond_monash_train_test_comparison.slurm`** ⭐ NEW
- **Purpose**: Compare train vs test performance on monthly datasets (sanity check)
- **Model Type**: Preconditioned checkpoint
- **Features**:
  - ✅ Evaluates BOTH train and test splits
  - ✅ Monthly datasets only (out-of-sample during pretraining)
  - ✅ Provides train/test gap analysis
  - ✅ Uses new `monash_cached_train` config
- **Usage**:
  ```bash
  sbatch eval_precond_monash_train_test_comparison.slurm
  ```
- **Status**: ✅ Just created, addresses train vs test comparison need

---

#### 3. **`eval_official_moirai_small.slurm`** ⭐ BASELINE
- **Purpose**: Evaluate official Salesforce Moirai-1.1-R-small baseline
- **Model Type**: Official HuggingFace pretrained model (no custom training)
- **Features**:
  - ✅ Downloads model from HuggingFace
  - ✅ No preconditioning (standard baseline)
  - ✅ Evaluates all frequencies
  - ✅ Good for comparison with your trained models
- **Usage**:
  ```bash
  sbatch eval_official_moirai_small.slurm
  ```
- **Status**: ✅ Keep for baseline comparisons

---

### ⚠️ **ARCHIVE - Redundant/Superseded Scripts**

#### 4. **`eval_moirai_monash_frequencies.slurm`**
- **Purpose**: Evaluate on all frequencies (yearly, quarterly, monthly)
- **Issue**:
  - ❌ Does NOT support preconditioning
  - ❌ Uses generic `moirai_lightning_ckpt` (not precond-aware)
  - ❌ Superseded by `eval_precond_monash.slurm`
- **Status**: 🗄️ ARCHIVE - functionally replaced
- **Recommendation**: Move to `archive/` folder

---

#### 5. **`eval_moirai_by_frequency.slurm`**
- **Purpose**: Evaluate single frequency at a time (parametric)
- **Issue**:
  - ❌ Does NOT support preconditioning
  - ❌ Less convenient than running all frequencies at once
  - ❌ Superseded by `eval_precond_monash.slurm` (which does all)
- **Status**: 🗄️ ARCHIVE - less useful now
- **Recommendation**: Move to `archive/` folder

---

#### 6. **`eval_moirai_checkpoint.slurm`**
- **Purpose**: Evaluate single dataset (most basic script)
- **Issue**:
  - ❌ Does NOT support preconditioning
  - ❌ Only evaluates one dataset at a time
  - ❌ Superseded by more comprehensive scripts
- **Status**: 🗄️ ARCHIVE - too basic, no preconditioning
- **Recommendation**: Move to `archive/` folder

---

## 📁 RECOMMENDED FILE STRUCTURE

```
/scratch/gpfs/EHAZAN/jh1161/
├── eval_precond_monash.slurm                        ⭐ KEEP - Primary eval script
├── eval_precond_monash_train_test_comparison.slurm  ⭐ KEEP - Train/test comparison
├── eval_official_moirai_small.slurm                 ⭐ KEEP - Baseline
│
└── archive/                                         🗄️ Archive old scripts
    ├── eval_moirai_monash_frequencies.slurm        (no preconditioning)
    ├── eval_moirai_by_frequency.slurm              (no preconditioning)
    └── eval_moirai_checkpoint.slurm                (no preconditioning)
```

---

## 🔑 KEY DIFFERENCES SUMMARY

| Script | Preconditioning | Datasets | Train/Test Split | Use Case |
|--------|----------------|----------|------------------|----------|
| `eval_precond_monash.slurm` | ✅ YES | All 12 | Test only | **Primary eval for preconditioned models** |
| `eval_precond_monash_train_test_comparison.slurm` | ✅ YES | Monthly (4) | BOTH | **Sanity check train vs test gap** |
| `eval_official_moirai_small.slurm` | ❌ NO | All 12 | Test only | **Baseline comparison** |
| ~~`eval_moirai_monash_frequencies.slurm`~~ | ❌ NO | All 12 | Test only | Obsolete - no precond support |
| ~~`eval_moirai_by_frequency.slurm`~~ | ❌ NO | By freq | Test only | Obsolete - less convenient |
| ~~`eval_moirai_checkpoint.slurm`~~ | ❌ NO | Single | Test only | Obsolete - too basic |

---

## 🚀 RECOMMENDED WORKFLOW

### Step 1: Evaluate Preconditioned Model (Test Set)
```bash
sbatch eval_precond_monash.slurm
```
This runs evaluation on all 12 Monash datasets (test split only).

### Step 2: Sanity Check (Train vs Test)
```bash
sbatch eval_precond_monash_train_test_comparison.slurm
```
This evaluates monthly datasets on BOTH train and test splits to check generalization.

### Step 3: Baseline Comparison
```bash
sbatch eval_official_moirai_small.slurm
```
This runs the official Moirai model for comparison.

---

## 🐛 BUG STATUS

### Known Issues (FIXED):
- ✅ Preconditioning reversal now works properly
- ✅ Train split evaluation now supported via `get_gluonts_train_dataset()`
- ✅ Config file `monash_cached_train.yaml` created

### No Known Bugs:
All three KEEP scripts are working and bug-free.

---

## 📝 CONFIGURATION FILES

### Data Configs (in `uni2ts/cli/conf/eval/data/`):
- ✅ `monash_cached.yaml` - Test split evaluation
- ✅ `monash_cached_train.yaml` - Train split evaluation (NEW)

### Model Configs (in `uni2ts/cli/conf/eval/model/`):
- ✅ `moirai_precond_ckpt.yaml` - Preconditioned checkpoint loading
- ✅ `moirai_lightning_ckpt.yaml` - Standard checkpoint (no precond)
- ✅ `moirai_1.1_R_small.yaml` - Official HuggingFace model

---

## 🎯 ACTION ITEMS

1. ✅ Keep 3 core scripts (precond, train_test, official)
2. ⏳ Create `archive/` directory
3. ⏳ Move 3 obsolete scripts to `archive/`
4. ✅ Document differences in this file
5. ⏳ (Optional) Add version/date tags to kept scripts

---

## 📞 QUICK REFERENCE

**Which script should I use?**

- Evaluating your preconditioned model? → `eval_precond_monash.slurm`
- Need train vs test comparison? → `eval_precond_monash_train_test_comparison.slurm`
- Need baseline comparison? → `eval_official_moirai_small.slurm`
- Have old non-preconditioned checkpoint? → Check `archive/` folder

---

*Last Updated: 2025-11-05*
*Location: /scratch/gpfs/EHAZAN/jh1161/EVAL_SCRIPTS_SUMMARY.md*
