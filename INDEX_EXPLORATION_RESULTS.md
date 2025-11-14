# Evaluation Pipeline and Preconditioning Exploration - Complete Results

**Date**: 2025-11-02  
**Status**: COMPLETE - Very Thorough Analysis  
**Working Directory**: `/scratch/gpfs/EHAZAN/jh1161`

---

## DOCUMENTATION GENERATED

This exploration produced three comprehensive documents:

### 1. **EVALUATION_PRECONDITIONING_ANALYSIS.md** (Main Technical Report)
   - **Size**: 26 KB, ~600 lines
   - **Purpose**: Complete architectural analysis
   - **Contents**:
     - Executive summary of the core problem
     - Detailed evaluation scripts analysis (eval.py, evaluation.py, data.py)
     - Complete preconditioning implementation breakdown
       - PolynomialPrecondition class analysis
       - ReversePrecondition class analysis
     - Moirai forecast models structure
       - MoiraiForecast (standard, incomplete)
       - MoiraiForecastPrecond (attempted fix)
     - Training pipeline reference (showing how it should work)
     - Data pipeline flow and transformation chain
     - Evaluation configuration file analysis
     - PyTorchPredictor integration issues
     - Problem summary and needed fixes (4 critical issues)
     - Implementation strategy (short-term and medium-term)
     - Key files and functions reference table
     - Data structures and transformation details
   - **Best For**: Deep understanding of the architecture and implementing fixes

### 2. **PRECONDITIONING_QUICK_REFERENCE.md** (Quick Lookup Guide)
   - **Size**: 12 KB, ~300 lines
   - **Purpose**: Quick reference for developers
   - **Contents**:
     - Core problem in 30 seconds
     - File locations at a glance (directory tree)
     - Key classes and methods (PolynomialPrecondition, ReversePrecondition, MoiraiForecast, MoiraiForecastPrecond)
     - Evaluation pipeline entry point
     - Data flow diagram
     - Training vs Evaluation comparison
     - What needs to change (4 changes required)
     - Mathematical formulas for preconditioning/reversal
     - Debugging checklist
     - Quick bash commands for exploration
   - **Best For**: Quick lookup while implementing fixes, debugging

### 3. **EXPLORATION_SUMMARY.txt** (Executive Summary)
   - **Size**: 11 KB, ~200 lines
   - **Purpose**: High-level overview and findings
   - **Contents**:
     - Evaluation scripts identified
     - Preconditioning implementation status
     - Moirai forecast class structure
     - Data pipeline and transformations
     - Training vs evaluation mismatch (the core issue)
     - Evaluation configuration files status
     - Key findings and issues (5 identified)
     - What needs to be fixed (4 fix levels)
     - Files affected by needed changes
     - Conclusion with impact assessment
   - **Best For**: Understanding the problem at a high level, presenting to others

---

## QUICK PROBLEM SUMMARY

**The Core Issue**:
```
TRAINING: Data → PolynomialPrecondition → Standardize → Model
EVALUATION: Data → Standardize → Model (NO preconditioning!)
```

**Result**: Models trained with preconditioning are evaluated without it, creating an unfair comparison.

**Root Cause**: `MoiraiForecast.get_default_transform()` doesn't include `PolynomialPrecondition`, even when the model was trained with it enabled.

---

## KEY FINDINGS

### What's Working Well
- PolynomialPrecondition class: Fully implemented, vectorized
- ReversePrecondition class: Fully implemented, handles 1D/2D/3D
- Training pipeline: Correctly applies preconditioning
- Transformation framework: Chain operations work properly

### What's Broken
1. **Input Transform Missing**: `get_default_transform()` doesn't include PolynomialPrecondition
2. **No Output Transform**: PyTorchPredictor can't apply ReversePrecondition
3. **MoiraiForecastPrecond Incomplete**: forward() method bypasses preconditioning
4. **Config Parameters Unused**: Preconditioning params in config but not used
5. **PyTorchPredictor Limitation**: No output_transform parameter support

### What Needs Fixing

**Priority 1 (Critical)**:
- Modify `MoiraiForecast.get_default_transform()` to include PolynomialPrecondition
- Pass preconditioning parameters through pipeline
- Apply ReversePrecondition after predictions

**Priority 2 (Important)**:
- Create custom predictor or wrapper with output_transform support
- Debug and fix MoiraiForecastPrecond.forward() method
- Update configuration files

---

## FILE STRUCTURE OF ANALYSIS

```
All files located in: /scratch/gpfs/EHAZAN/jh1161/

📊 Documentation:
├── EVALUATION_PRECONDITIONING_ANALYSIS.md    (Main technical report)
├── PRECONDITIONING_QUICK_REFERENCE.md        (Quick reference)
├── EXPLORATION_SUMMARY.txt                   (Executive summary)
└── INDEX_EXPLORATION_RESULTS.md              (This file)

📁 Source Code Referenced:
└── uni2ts/
    ├── cli/
    │   ├── eval.py                          (Evaluation entry point)
    │   └── conf/eval/
    │       ├── default.yaml
    │       ├── model/moirai_lightning_ckpt_precond.yaml
    │       └── data/monash.yaml
    ├── src/uni2ts/
    │   ├── transform/
    │   │   ├── precondition.py              (Core implementations)
    │   │   └── _base.py                     (Transformation base)
    │   ├── model/moirai/
    │   │   ├── forecast.py                  (Standard model - has issue)
    │   │   ├── forecast_precond.py          (Attempted fix - incomplete)
    │   │   └── pretrain.py                  (Training - reference)
    │   └── eval_util/
    │       ├── evaluation.py                (Evaluation functions)
    │       └── data.py                      (Data loading)
```

---

## HOW TO USE THESE DOCUMENTS

### For Understanding the Problem
1. Start with **EXPLORATION_SUMMARY.txt** - 5 min read
2. Review section "5. TRAINING VS EVALUATION MISMATCH"
3. Look at section "7. KEY FINDINGS AND ISSUES"

### For Implementing Fixes
1. Read **EVALUATION_PRECONDITIONING_ANALYSIS.md** sections 3 and 8
2. Use **PRECONDITIONING_QUICK_REFERENCE.md** as you code
3. Follow the "WHAT NEEDS TO CHANGE" section
4. Use the debugging checklist while testing

### For Quick Lookups During Development
1. Use **PRECONDITIONING_QUICK_REFERENCE.md**
2. Check file locations in directory tree
3. Review mathematical formulas
4. Use bash commands for code exploration
5. Follow debugging checklist

### For Presenting to Others
1. Use **EXPLORATION_SUMMARY.txt** as overview
2. Show the core problem diagram
3. Explain the 5 key issues
4. Walk through training vs evaluation flow

---

## CRITICAL CODE LOCATIONS

### The Main Problem (Line Numbers)
- **Missing Preconditioning**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast.py:941`
  - Function: `get_default_transform()`
  - Issue: Doesn't include `PolynomialPrecondition`

- **No Reversal**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/eval_util/evaluation.py:228`
  - Function: `evaluate_model()`
  - Issue: No `ReversePrecondition` applied

- **Incomplete Fix**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast_precond.py:204`
  - Function: `forward()`
  - Issue: Bypasses preconditioning

### The Reference Implementation
- **Training With Precondition**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/pretrain.py:387-392`
  - Shows how it SHOULD work
  - Reference for implementing fixes

### The Well-Implemented Components
- **PolynomialPrecondition**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py:24-248`
- **ReversePrecondition**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py:250-376`

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Add Input Preconditioning
- [ ] Modify `MoiraiForecast.__init__()` to accept preconditioning parameters
- [ ] Store parameters in `self.hparams`
- [ ] Modify `get_default_transform()` to include PolynomialPrecondition
- [ ] Pass parameters from config to model

### Phase 2: Add Output Reversal
- [ ] Create wrapper class or custom predictor for output_transform
- [ ] Apply ReversePrecondition after predictions
- [ ] Update `evaluate_model()` to handle reversal
- [ ] Test metric computation in correct space

### Phase 3: Fix MoiraiForecastPrecond
- [ ] Debug forward() method "shape mismatch" issue
- [ ] Test preconditioning with standardization
- [ ] Verify iterative reversal works correctly
- [ ] Make it work as primary preconditioning model

### Phase 4: Configuration
- [ ] Update model configs to include preconditioning parameters
- [ ] Create separate configs for preconditioned models
- [ ] Test end-to-end evaluation pipeline

---

## MATHEMATICAL REFERENCE (Quick)

### Preconditioning (Forward)
```
ỹₜ = yₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  (for t > n)
```
- Applied before training
- Removes auto-regressive structure
- Improves condition number of hidden transition matrices

### Reversal (Inverse)
```
yₜ = ỹₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  (for t > n)
```
- Computed iteratively (each value depends on previous)
- Restores original space for metrics
- Applied after predictions

---

## RELATED DOCUMENTATION

**Also See**:
- `/scratch/gpfs/EHAZAN/jh1161/EVALUATION_GUIDE.md` - Evaluation procedures
- `/scratch/gpfs/EHAZAN/jh1161/QUICKSTART_EVALUATION.md` - Quick start
- Training configuration in project directories

---

## NEXT STEPS

1. **Read EXPLORATION_SUMMARY.txt** for overview (5 minutes)
2. **Read EVALUATION_PRECONDITIONING_ANALYSIS.md** sections 1-5 for details (15 minutes)
3. **Use PRECONDITIONING_QUICK_REFERENCE.md** while coding
4. **Follow implementation checklist** above
5. **Test thoroughly** with preconditioned models
6. **Verify metrics** are computed in correct space

---

## STATISTICS

**Files Analyzed**: 6 main files, 20+ supporting files
**Lines of Code Reviewed**: 2000+
**Issues Identified**: 5 critical
**Fix Levels Recommended**: 4 (immediate, core, proper, configuration)
**Documentation Generated**: 49 KB total
**Analysis Depth**: Very Thorough - Complete Architecture

---

**Generated**: 2025-11-02  
**Status**: Ready for implementation  
**Last Document**: INDEX_EXPLORATION_RESULTS.md
