# Training Comparison Guide

This guide shows how to compare training runs (preconditioned vs non-preconditioned).

## Your Two Runs

1. **Non-preconditioned:** `pretrain_run_20251020_205126`
2. **Preconditioned:** `precond_default_20251102_102511`

---

## Option 1: TensorBoard (Interactive - Best for Exploration)

### Quick Start

```bash
cd /scratch/gpfs/EHAZAN/jh1161
./compare_training_runs.sh
```

Then access at: `http://localhost:6006`

### If on Cluster (SSH Tunnel)

**On your local machine:**
```bash
ssh -L 6006:localhost:6006 jh1161@della.princeton.edu
```

**Then on the cluster:**
```bash
cd /scratch/gpfs/EHAZAN/jh1161
./compare_training_runs.sh
```

**On your local browser:**
- Open: http://localhost:6006

### Manual TensorBoard Command

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

tensorboard --logdir_spec \
  "non-precond:outputs/pretrain/moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/logs,precond:outputs/pretrain/moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/logs" \
  --port 6006
```

### TensorBoard Features

- **Scalars tab**: View loss curves
- **Compare runs**: Toggle checkboxes to show/hide runs
- **Smoothing**: Adjust slider to smooth noisy curves
- **Download data**: Export CSV for further analysis
- **Zoom**: Click and drag to zoom into specific regions

---

## Option 2: Python Script (Static Plots - Best for Papers/Reports)

### Quick Start

```bash
cd /scratch/gpfs/EHAZAN/jh1161
python plot_training_comparison.py
```

### Output

Creates 4 plots:

1. **`training_comparison.png`** - Overview (train + val side-by-side)
2. **`training_plots/training_loss.png`** - Detailed training loss
3. **`training_plots/validation_loss.png`** - Detailed validation loss
4. **`training_plots/final_loss_comparison.png`** - Bar chart of final values

### Example Output

```
✓ Plot saved to: training_comparison.png
✓ Saved: training_plots/training_loss.png
✓ Saved: training_plots/validation_loss.png
✓ Saved: training_plots/final_loss_comparison.png

Final loss values:

Non-preconditioned:
  Training:   4.3100 (at step 100000)
  Validation: 4.3500 (at step 100000)

Preconditioned:
  Training:   4.0400 (at step 100000)
  Validation: 4.0800 (at step 100000)
  Improvement: 0.2700 (6.27%)
```

### Customizing the Script

Edit `plot_training_comparison.py` to:

**Add more runs:**
```python
runs = {
    "Non-preconditioned": (...),
    "Preconditioned": (...),
    "Your New Run": (
        "path/to/logs/version_0",
        "green",  # color
        "-."      # linestyle
    ),
}
```

**Change metrics:**
```python
metrics = [
    ("train/PackedNLLLoss", "Training Loss"),
    ("val/PackedNLLLoss", "Validation Loss"),
    # Add more metrics here
]
```

---

## Option 3: Quick Terminal Check

### View latest training loss

```bash
# Non-preconditioned
tail -50 /scratch/gpfs/EHAZAN/jh1161/logs/pretrain_1527008.out | grep "PackedNLLLoss"

# Preconditioned (find your log file)
tail -50 /scratch/gpfs/EHAZAN/jh1161/logs/pretrain_precond_*.out | grep "PackedNLLLoss"
```

---

## Comparison Checklist

When comparing training runs, check:

### 1. Final Loss Values
- [ ] Which model has lower training loss?
- [ ] Which model has lower validation loss?
- [ ] How large is the improvement?

### 2. Convergence Speed
- [ ] Which model converges faster?
- [ ] How many steps to reach 95% of final performance?

### 3. Stability
- [ ] Are loss curves smooth or noisy?
- [ ] Any sudden spikes or divergence?

### 4. Overfitting
- [ ] What's the train/val gap?
- [ ] Does validation loss start increasing while training decreases?

### 5. Training Time
- [ ] Iterations per second (it/s)
- [ ] Total wall-clock time
- [ ] GPU memory usage

---

## Interpreting Results

### Good Signs for Preconditioning

✓ **Lower final loss** (both train and val)
```
Non-precond: 4.31
Precond:     4.04  ← 0.27 improvement (6.3%)
```

✓ **Faster convergence**
```
Non-precond: 80k steps to reach 4.35
Precond:     50k steps to reach 4.35
```

✓ **Similar train/val gap** (no extra overfitting)
```
Non-precond: gap = 0.04 (4.35 - 4.31)
Precond:     gap = 0.04 (4.08 - 4.04)
```

✓ **Stable training**
- Smooth curves
- No divergence
- Consistent improvement

### Warning Signs

✗ **Higher validation loss** (despite lower training loss)
```
Non-precond: train=4.31, val=4.35
Precond:     train=4.00, val=4.50  ← Overfitting!
```

✗ **Slower convergence**
- Takes more steps to reach same performance

✗ **Training instability**
- Large spikes
- Oscillations
- Divergence

✗ **Negligible improvement**
```
Non-precond: 4.310
Precond:     4.305  ← Only 0.005 difference (not significant)
```

---

## Advanced: Extract Raw Data

### Using Python

```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator("path/to/logs/version_0")
ea.Reload()

# Get training loss
train_loss = ea.Scalars("train/PackedNLLLoss")
steps = [e.step for e in train_loss]
values = [e.value for e in train_loss]

# Save to CSV
import pandas as pd
df = pd.DataFrame({"step": steps, "loss": values})
df.to_csv("training_loss.csv", index=False)
```

### Using TensorBoard Web UI

1. Open TensorBoard
2. Click on "Scalars" tab
3. Hover over a run
4. Click the "..." menu
5. Select "Export as CSV"

---

## Quick Reference

| Task | Command |
|------|---------|
| **Interactive comparison** | `./compare_training_runs.sh` |
| **Static plots** | `python plot_training_comparison.py` |
| **Check final loss** | `tail logs/*.out \| grep PackedNLLLoss` |
| **View in TensorBoard** | `tensorboard --logdir outputs/pretrain` |
| **SSH tunnel (local)** | `ssh -L 6006:localhost:6006 user@host` |

---

## Troubleshooting

### "No event files found"

**Problem:** Script can't find TensorBoard logs

**Solution:**
```bash
# Check if logs exist
ls -la /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/*/logs/version_0/

# If missing, check run directory
ls -la /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/
```

### "Module 'tensorboard' not found"

**Problem:** TensorBoard not installed in your environment

**Solution:**
```bash
source venv/bin/activate
pip install tensorboard matplotlib
```

### "Port 6006 already in use"

**Problem:** Another TensorBoard instance is running

**Solution:**
```bash
# Use a different port
tensorboard --logdir ... --port 6007

# Or kill existing instance
pkill -f tensorboard
```

### Plots don't show improvement

**Possible reasons:**
1. Preconditioning didn't help (valid result!)
2. Training not complete (check if runs finished)
3. Different number of training steps
4. Different hyperparameters

**Check:**
```bash
# Compare training logs
tail -100 logs/pretrain_*.out
```

---

## Summary

**Easiest:** TensorBoard for quick interactive exploration
```bash
./compare_training_runs.sh
```

**Best for reports:** Python script for publication-quality plots
```bash
python plot_training_comparison.py
```

Both methods let you see:
- Training loss curves
- Validation loss curves
- Convergence speed
- Final performance

Lower loss = better model! 🎉
