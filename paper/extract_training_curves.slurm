#!/bin/bash
#SBATCH --job-name=tb_curves
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/tb_curves_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/tb_curves_%j.err

module load anaconda3/2024.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

python3 << 'PYEOF'
"""Extract FULL training loss curves from ALL TensorBoard logs."""
import os, glob, json, csv
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2"
OUT_DIR = "/scratch/gpfs/EHAZAN/jh1161/paper/training_curves"
os.makedirs(OUT_DIR, exist_ok=True)

# Key runs to extract full curves for
KEY_RUNS = [
    # Main 5-seed comparisons (10K)
    "baseline_m2d_seed0_10k", "baseline_m2d_seed1_10k", "baseline_m2d_seed2_10k",
    "baseline_m2d_seed7_10k", "baseline_m2d_seed42_10k",
    "hd10_m2d_seed0_10k", "hd10_m2d_seed1_10k", "hd10_m2d_seed2_10k",
    "hd10_m2d_seed7_10k", "hd10_m2d_seed42_10k",
    "zero_m2d_seed0_10k", "zero_m2d_seed1_10k", "zero_m2d_seed2_10k",
    "zero_m2d_seed7_10k", "zero_m2d_seed42_10k",
    "duplicate_m2d_seed0_10k", "duplicate_m2d_seed1_10k", "duplicate_m2d_seed2_10k",
    "duplicate_m2d_seed7_10k", "duplicate_m2d_seed42_10k",
    # HD0 (no dropout) 5-seed
    "hd0_m2d_seed0_10k", "hd0_m2d_seed1_10k", "hd0_m2d_seed2_10k",
    "hd0_m2d_seed7_10k", "hd0_m2d_seed42_10k",
    # MS46 5-seed
    "ms46_m2d_seed0_10k", "ms46_m2d_seed1_10k", "ms46_m2d_seed2_10k",
    "ms46_m2d_seed7_10k", "ms46_m2d_seed42_10k",
    # MSHD10
    "mshd10_m2d_seed0_10k", "mshd10_m2d_seed1_10k", "mshd10_m2d_seed2_10k",
    "ms46hd10_m2d_seed0_10k", "ms46hd10_m2d_seed7_10k", "ms46hd10_m2d_seed42_10k",
    # 100K runs
    "baseline_m2d_s0_100k", "baseline_m2d_s1_100k", "baseline_m2d_s2_100k",
    "baseline_m2d_s7_100k", "baseline_m2d_s42_100k",
    "hd10_m2d_s0_100k", "hd10_m2d_s1_100k", "hd10_m2d_s2_100k",
    "hd10_m2d_s7_100k", "hd10_m2d_s42_100k",
    "mshd10_m2d_s0_100k", "mshd10_m2d_s2_100k", "mshd10_m2d_s7_100k",
    "duplicate_m2d_s0_100k", "duplicate_m2d_s1_100k", "duplicate_m2d_s2_100k",
    "duplicate_m2d_s7_100k", "duplicate_m2d_s42_100k",
    "zero_m2d_s0_100k", "zero_m2d_s1_100k", "zero_m2d_s2_100k",
    "zero_m2d_s7_100k", "zero_m2d_s42_100k",
    # LR=2e-3
    "baseline_lr2e3_s0_10k", "baseline_lr2e3_seed1_10k", "baseline_lr2e3_seed2_10k",
    "baseline_lr2e3_seed7_10k", "baseline_lr2e3_seed42_10k",
    "hd10_lr2e3_s0_10k", "hd10_lr2e3_seed1_10k", "hd10_lr2e3_seed2_10k",
    "hd10_lr2e3_seed7_10k", "hd10_lr2e3_seed42_10k",
    # LR control 100K
    "bl_cosine_floor10_s0_100k", "bl_wsd_s0_100k",
    "hd10_cosine_floor10_s0_100k", "hd10_wsd_s0_100k",
    # Ablations (seed 0)
    "hd2_m2d_seed0_10k", "hd3_m2d_seed0_10k", "hd5_m2d_seed0_10k",
    "hd6_m2d_seed0_10k", "hd7_m2d_seed0_10k", "hd8_m2d_seed0_10k",
    "hd10_s4_m2d_seed0_10k", "hd10_s8_m2d_seed0_10k", "hd10_s32_m2d_seed0_10k",
    "ema_hint_m2d_seed0_10k", "leg4_hd10_m2d_seed0_10k", "leg6_hd10_m2d_seed0_10k",
    "diff4_hd10_m2d_seed0_10k",
    "learned_hd10_m2d_seed0_10k", "learned_zero_hd10_m2d_seed0_10k",
    "hd10_drop20_m2d_seed0_10k", "hd10sd20_m2d_seed0_10k",
    "hd30_m2d_seed0_10k", "sd30_m2d_seed0_10k",
    # Latent
    "lat_c4s1_m2d_seed0_10k", "lat_c4s4_m2d_seed0_10k", "lat_c6s1_m2d_seed0_10k",
    # Multi-stride
    "multi_stride_m2d_seed0_10k",
    # Degree multi-seed (new)
    "hd2_m2d_seed0_10k",
]

# Also get ALL runs for completeness
all_runs = sorted(os.listdir(BASE))

all_curves = {}

for run_name in all_runs:
    run_dir = os.path.join(BASE, run_name)
    if not os.path.isdir(run_dir):
        continue

    log_dirs = sorted(glob.glob(os.path.join(run_dir, "logs/version_*")))
    if not log_dirs:
        continue

    try:
        ea = EventAccumulator(log_dirs[-1])
        ea.Reload()

        curve_data = {}
        for tag in ea.Tags().get('scalars', []):
            if 'loss' in tag.lower() or 'lr' in tag.lower():
                events = ea.Scalars(tag)
                curve_data[tag] = {
                    'steps': [e.step for e in events],
                    'values': [e.value for e in events],
                    'wall_times': [e.wall_time for e in events],
                }

        if curve_data:
            all_curves[run_name] = curve_data
            n_points = max(len(v['steps']) for v in curve_data.values())
            print(f"  {run_name}: {len(curve_data)} tags, {n_points} points")

    except Exception as e:
        print(f"  {run_name}: FAILED ({e})")

# Save all curves as JSON
with open(os.path.join(OUT_DIR, "all_curves.json"), 'w') as f:
    json.dump(all_curves, f)
print(f"\nSaved {len(all_curves)} run curves to {OUT_DIR}/all_curves.json")

# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "="*80)
print("TRAINING DYNAMICS ANALYSIS")
print("="*80)

epoch_tag = "train/PackedQuantileMAELoss_epoch"
step_tag = "train/PackedQuantileMAELoss_step"

def get_loss_curve(run_name, prefer_epoch=True):
    if run_name not in all_curves:
        return None, None
    tags = all_curves[run_name]
    tag = epoch_tag if (prefer_epoch and epoch_tag in tags) else step_tag if step_tag in tags else None
    if tag is None:
        return None, None
    return np.array(tags[tag]['steps']), np.array(tags[tag]['values'])

# 1. CONVERGENCE SPEED: When does each method reach 90% of final improvement?
print("\n--- 1. CONVERGENCE SPEED (10K runs, seed 0) ---")
methods_10k = {
    'Baseline': 'baseline_m2d_seed0_10k',
    'HD10': 'hd10_m2d_seed0_10k',
    'Zero': 'zero_m2d_seed0_10k',
    'Duplicate': 'duplicate_m2d_seed0_10k',
    'HD0': 'hd0_m2d_seed0_10k',
    'MS46': 'ms46_m2d_seed0_10k',
}
for name, run in methods_10k.items():
    steps, vals = get_loss_curve(run)
    if steps is None:
        continue
    initial = vals[0]
    final = vals[-1]
    improvement = initial - final
    target_90 = initial - 0.9 * improvement
    reached_90 = None
    for i, v in enumerate(vals):
        if v <= target_90:
            reached_90 = steps[i]
            break
    print(f"  {name:15s}: initial={initial:.4f} final={final:.4f} Δ={improvement:.4f} 90%@step={reached_90}")

# 2. EARLY TRAINING: First 1000 steps comparison
print("\n--- 2. EARLY TRAINING (first 1000 steps, step-level) ---")
for name, run in methods_10k.items():
    steps, vals = get_loss_curve(run, prefer_epoch=False)
    if steps is None or len(steps) < 10:
        continue
    # Find loss at step ~1000
    mask = steps <= 1000
    if mask.sum() > 0:
        early_steps = steps[mask]
        early_vals = vals[mask]
        print(f"  {name:15s}: step[0]={early_vals[0]:.4f} step[~1000]={early_vals[-1]:.4f} Δ={early_vals[0]-early_vals[-1]:.4f}")

# 3. LOSS CURVE SMOOTHNESS (variance of epoch-to-epoch changes)
print("\n--- 3. LOSS CURVE SMOOTHNESS (10K, all seeds) ---")
for method_name, prefix in [('Baseline', 'baseline_m2d_seed'), ('HD10', 'hd10_m2d_seed'),
                              ('Zero', 'zero_m2d_seed'), ('Duplicate', 'duplicate_m2d_seed')]:
    all_volatilities = []
    for seed in ['0', '1', '2', '7', '42']:
        run = f"{prefix}{seed}_10k"
        steps, vals = get_loss_curve(run)
        if vals is not None and len(vals) > 5:
            diffs = np.diff(vals)
            volatility = np.std(diffs) / np.mean(np.abs(vals[:-1]))
            all_volatilities.append(volatility)
    if all_volatilities:
        print(f"  {method_name:15s}: mean_volatility={np.mean(all_volatilities):.6f} ± {np.std(all_volatilities):.6f}")

# 4. 100K DYNAMICS: Loss trajectory and collapse detection
print("\n--- 4. 100K TRAINING DYNAMICS ---")
for method_name, prefix in [('BL_100K', 'baseline_m2d_s'), ('HD10_100K', 'hd10_m2d_s'),
                              ('Dup_100K', 'duplicate_m2d_s'), ('Zero_100K', 'zero_m2d_s')]:
    for seed in ['0', '1', '2', '7', '42']:
        run = f"{prefix}{seed}_100k"
        steps, vals = get_loss_curve(run)
        if vals is not None and len(vals) > 10:
            # Check for loss spikes (>2 std from rolling mean)
            window = min(50, len(vals)//4)
            if window > 2:
                rolling_mean = np.convolve(vals, np.ones(window)/window, mode='valid')
                rolling_std = np.array([np.std(vals[max(0,i-window):i+1]) for i in range(len(vals))])
                # Find epochs where loss increases significantly
                diffs = np.diff(vals)
                spikes = np.where(diffs > 2 * np.std(diffs))[0]
                min_loss_epoch = np.argmin(vals)
                final_vs_min = (vals[-1] - vals[min_loss_epoch]) / vals[min_loss_epoch] * 100

                print(f"  {method_name} s{seed}: final={vals[-1]:.6f} min={vals[min_loss_epoch]:.6f}@ep{steps[min_loss_epoch]} "
                      f"final_vs_min={final_vs_min:+.2f}% spikes={len(spikes)}")

# 5. HD10 vs BL GAP OVER TIME
print("\n--- 5. HD10 vs BL LOSS GAP OVER TRAINING (seed 0, 10K) ---")
bl_steps, bl_vals = get_loss_curve('baseline_m2d_seed0_10k')
hd_steps, hd_vals = get_loss_curve('hd10_m2d_seed0_10k')
if bl_vals is not None and hd_vals is not None:
    min_len = min(len(bl_vals), len(hd_vals))
    for i in [0, min_len//4, min_len//2, 3*min_len//4, min_len-1]:
        gap = (bl_vals[i] - hd_vals[i]) / bl_vals[i] * 100
        print(f"  Epoch {bl_steps[i]:>5d}: BL={bl_vals[i]:.6f} HD10={hd_vals[i]:.6f} gap={gap:+.3f}%")

# 6. HD10 vs BL GAP OVER TIME (100K)
print("\n--- 6. HD10 vs BL LOSS GAP OVER TRAINING (seed 0, 100K) ---")
bl_steps, bl_vals = get_loss_curve('baseline_m2d_s0_100k')
hd_steps, hd_vals = get_loss_curve('hd10_m2d_s0_100k')
if bl_vals is not None and hd_vals is not None:
    min_len = min(len(bl_vals), len(hd_vals))
    checkpoints = [0, 99, 199, 299, 499, 699, 899, min_len-1]
    for i in checkpoints:
        if i < min_len:
            gap = (bl_vals[i] - hd_vals[i]) / bl_vals[i] * 100
            print(f"  Epoch {bl_steps[i]:>5d}: BL={bl_vals[i]:.6f} HD10={hd_vals[i]:.6f} gap={gap:+.3f}%")

# 7. LR SCHEDULE COMPARISON (100K)
print("\n--- 7. LR SCHEDULE COMPARISON (100K, seed 0) ---")
for run_name in ['baseline_m2d_s0_100k', 'bl_cosine_floor10_s0_100k', 'bl_wsd_s0_100k',
                  'hd10_m2d_s0_100k', 'hd10_cosine_floor10_s0_100k', 'hd10_wsd_s0_100k']:
    steps, vals = get_loss_curve(run_name)
    if vals is not None:
        min_idx = np.argmin(vals)
        print(f"  {run_name:40s}: final={vals[-1]:.6f} min={vals[min_idx]:.6f}@ep{steps[min_idx]} "
              f"n_epochs={len(vals)}")

# 8. LEARNING RATE CURVES
print("\n--- 8. LEARNING RATE CURVES ---")
for run_name in ['baseline_m2d_seed0_10k', 'hd10_m2d_seed0_10k',
                  'baseline_m2d_s0_100k', 'hd10_m2d_s0_100k']:
    if run_name in all_curves:
        lr_tags = [t for t in all_curves[run_name] if 'lr' in t.lower()]
        for tag in lr_tags:
            data = all_curves[run_name][tag]
            if data['values']:
                print(f"  {run_name}: {tag} initial={data['values'][0]:.6f} peak={max(data['values']):.6f} final={data['values'][-1]:.8f}")

# 9. CROSS-SEED VARIANCE OVER TIME (10K)
print("\n--- 9. CROSS-SEED LOSS VARIANCE OVER TIME (10K) ---")
for method_name, prefix in [('BL', 'baseline_m2d_seed'), ('HD10', 'hd10_m2d_seed')]:
    seed_curves = {}
    for seed in ['0', '1', '2', '7', '42']:
        run = f"{prefix}{seed}_10k"
        steps, vals = get_loss_curve(run)
        if vals is not None:
            seed_curves[seed] = vals

    if len(seed_curves) >= 3:
        min_len = min(len(v) for v in seed_curves.values())
        arrays = np.array([v[:min_len] for v in seed_curves.values()])
        stds = np.std(arrays, axis=0)
        means = np.mean(arrays, axis=0)
        cvs = stds / means
        print(f"  {method_name}: CV at epoch 0={cvs[0]:.6f}, epoch 50={cvs[min(50, min_len-1)]:.6f}, "
              f"final={cvs[-1]:.6f}, max_CV={np.max(cvs):.6f}@epoch={np.argmax(cvs)}")

# 10. LATENT PRECONDITIONING DYNAMICS
print("\n--- 10. LATENT PRECONDITIONING TRAINING DYNAMICS ---")
for run in ['lat_c4s1_m2d_seed0_10k', 'lat_c4s4_m2d_seed0_10k', 'lat_c6s1_m2d_seed0_10k']:
    steps, vals = get_loss_curve(run)
    if vals is not None:
        print(f"  {run}: initial={vals[0]:.4f} ep10={vals[min(10, len(vals)-1)]:.4f} "
              f"ep50={vals[min(50, len(vals)-1)]:.4f} final={vals[-1]:.4f}")

# 11. DEGREE ABLATION CONVERGENCE
print("\n--- 11. DEGREE ABLATION CONVERGENCE (seed 0, 10K) ---")
for d in [2, 3, 4, 5, 6, 7, 8]:
    run = f"hd{d}_m2d_seed0_10k" if d != 4 else "hd10_m2d_seed0_10k"
    steps, vals = get_loss_curve(run)
    if vals is not None:
        print(f"  d={d}: initial={vals[0]:.4f} ep25={vals[min(25, len(vals)-1)]:.4f} "
              f"ep50={vals[min(50, len(vals)-1)]:.4f} final={vals[-1]:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
PYEOF
