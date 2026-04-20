#!/usr/bin/env python3
"""
Training convergence analysis: Baseline vs hd10 (hint d=4 + 10% dropout).

Plots GIFT-Eval geometric mean MASE vs training steps for both models,
with improvement shading, best-so-far markers, convergence speed annotation,
and smoothed training loss inset.

Data sources:
  - Baseline: m2_baseline_20260209_114203 (10K warmup, 100K steps)
  - hd10: m2_hd10_100k_20260223_112829 (1K warmup, 100K steps)
  - ms46 short-run: q_ms_d4d6_20260223_134653 (1K warmup, 10K steps)
  - ms46+drop 100K: ms_d4d6_100k_drop_20260227_012355 (10K warmup, 100K steps)
  - Training loss: analysis/baseline_vs_best100k/training_loss_data.json

Usage:
    python analysis/convergence_analysis.py
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- output path ----------
FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- publication style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral", "cmr10"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
})

# ---------- MASE data (verified from report files) ----------

# Baseline (10K warmup, cosine annealing, 100K total)
# Source: m2_baseline_20260209_114203
baseline_steps = np.array([10, 20, 30, 50, 70, 100])  # in thousands
baseline_mase = np.array([1.2421, 1.3025, 1.3127, 1.2784, 1.2780, 1.2911])

# hd10 (hint d=4 + 10% dropout, 1K warmup, cosine, 100K total)
# Source: m2_hd10_100k_20260223_112829
hd10_steps = np.array([20, 30, 50, 70, 100])  # in thousands
hd10_mase = np.array([1.2401, 1.2064, 1.2412, 1.2241, 1.1918])

# ms46 short-run (multi-scale d=4+d=6, 1K warmup, 10K total)
# Source: q_ms_d4d6_20260223_134653
ms46_short_steps = np.array([8, 10])  # in thousands
ms46_short_mase = np.array([1.2223, 1.1675])

# ms46 + 10% dropout, 100K run (10K warmup)
# Source: ms_d4d6_100k_drop_20260227_012355
ms46drop_steps = np.array([50, 100])  # in thousands
ms46drop_mase = np.array([1.2363, 1.2214])

# hd10 at 10K (short run, q_hint_drop10, 1K warmup, 10K total)
hd10_short_step = 10
hd10_short_mase = 1.1802


# ---------- helper: running best ----------
def running_best(steps, mases):
    """Return (steps, best_mase_so_far) arrays."""
    best = np.inf
    out = []
    for m in mases:
        best = min(best, m)
        out.append(best)
    return np.array(out)


# ---------- main plot ----------
def main():
    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Colors
    c_bl = "#4C72B0"     # blue
    c_hd = "#C44E52"     # red
    c_ms = "#55A868"     # green
    c_msd = "#8172B2"    # purple

    # --- Plot baseline ---
    ax.plot(baseline_steps, baseline_mase, "o-", color=c_bl, label="Baseline (10K warmup)",
            zorder=3)

    # --- Plot hd10 100K run ---
    ax.plot(hd10_steps, hd10_mase, "s-", color=c_hd,
            label="Hint $d{=}4$ + 10% drop (1K warmup)", zorder=3)
    # Also add hd10 short-run point at 10K
    ax.plot(hd10_short_step, hd10_short_mase, "s", color=c_hd,
            markeredgecolor="black", markeredgewidth=0.8, zorder=4)

    # --- Plot ms46 short-run ---
    ax.plot(ms46_short_steps, ms46_short_mase, "D-", color=c_ms,
            label="MS $d{=}4{+}d{=}6$ (1K warmup, 10K run)", zorder=3)

    # --- Plot ms46+drop 100K ---
    ax.plot(ms46drop_steps, ms46drop_mase, "^-", color=c_msd,
            label="MS $d{=}4{+}d{=}6$ + 10% drop (10K warmup)", zorder=3)

    # --- Improvement shading between baseline and hd10 ---
    # Interpolate hd10 to baseline steps for fill_between.
    # hd10 starts at 20K, so we only shade from 20K onwards.
    shared_steps = baseline_steps[baseline_steps >= 20]
    bl_shared = baseline_mase[baseline_steps >= 20]
    hd_shared = np.interp(shared_steps, hd10_steps, hd10_mase)
    ax.fill_between(shared_steps, bl_shared, hd_shared,
                    alpha=0.12, color=c_hd, label="_nolegend_")

    # --- Mark "best so far" for each model ---
    # Baseline best
    bl_best_idx = np.argmin(baseline_mase)
    bl_best_step = baseline_steps[bl_best_idx]
    bl_best_val = baseline_mase[bl_best_idx]
    ax.plot(bl_best_step, bl_best_val, "o", color=c_bl, markersize=12,
            markeredgecolor="black", markeredgewidth=1.5, fillstyle="none", zorder=5)

    # hd10 best (combine short-run and long-run)
    all_hd10_steps = np.concatenate([[hd10_short_step], hd10_steps])
    all_hd10_mase = np.concatenate([[hd10_short_mase], hd10_mase])
    hd_best_idx = np.argmin(all_hd10_mase)
    hd_best_step = all_hd10_steps[hd_best_idx]
    hd_best_val = all_hd10_mase[hd_best_idx]
    ax.plot(hd_best_step, hd_best_val, "s", color=c_hd, markersize=12,
            markeredgecolor="black", markeredgewidth=1.5, fillstyle="none", zorder=5)

    # ms46 best
    ms_best_idx = np.argmin(ms46_short_mase)
    ms_best_step = ms46_short_steps[ms_best_idx]
    ms_best_val = ms46_short_mase[ms_best_idx]
    ax.plot(ms_best_step, ms_best_val, "D", color=c_ms, markersize=12,
            markeredgecolor="black", markeredgewidth=1.5, fillstyle="none", zorder=5)

    # --- Convergence speed annotation ---
    # At what step does hd10 first beat the baseline's best-ever MASE?
    # Baseline best = 1.2421 at 10K
    # hd10 at 10K (short run) = 1.1802 < 1.2421 -- beats it immediately
    # hd10 100K run: 20K = 1.2401 < 1.2421 -- also beats it
    # So hd10 beats baseline best at its first evaluated checkpoint (10K or 20K)
    convergence_step = hd10_short_step  # 10K
    convergence_val = hd10_short_mase   # 1.1802

    ax.annotate(
        f"hd10 beats baseline best\nat {convergence_step}K steps\n(MASE {convergence_val:.4f} vs {bl_best_val:.4f})",
        xy=(convergence_step, convergence_val),
        xytext=(30, 1.205),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2,
                        connectionstyle="arc3,rad=0.3"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.7", alpha=0.9),
    )

    # --- Annotate ms46 best ---
    ax.annotate(
        f"Best overall: {ms_best_val:.4f}\n({(ms_best_val - bl_best_val) / bl_best_val * 100:.1f}% vs baseline)",
        xy=(ms_best_step, ms_best_val),
        xytext=(22, 1.145),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.7", alpha=0.9),
    )

    # --- Annotate hd10 100K best ---
    ax.annotate(
        f"hd10 100K: {hd_best_val:.4f}\n({(hd_best_val - bl_best_val) / bl_best_val * 100:.1f}% vs baseline)",
        xy=(hd_best_step, hd_best_val),
        xytext=(75, 1.155),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2,
                        connectionstyle="arc3,rad=-0.15"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.7", alpha=0.9),
    )

    # --- Dashed line for baseline best ---
    ax.axhline(y=bl_best_val, color=c_bl, linestyle=":", linewidth=1.0,
               alpha=0.5, zorder=1)
    ax.text(102, bl_best_val + 0.002, f"Baseline best = {bl_best_val:.4f}",
            fontsize=7, color=c_bl, va="bottom", ha="left")

    # --- Labels and formatting ---
    ax.set_xlabel("Training Steps ($\\times 10^3$)")
    ax.set_ylabel("Geometric Mean MASE (97 GIFT-Eval configs)")
    ax.set_title("Training Convergence: Baseline vs Hint Preconditioning")

    ax.set_xlim(5, 108)
    ax.set_ylim(1.14, 1.33)
    ax.set_xticks([10, 20, 30, 50, 70, 100])
    ax.set_xticklabels(["10K", "20K", "30K", "50K", "70K", "100K"])

    # Custom legend with "best" marker explanation
    legend_handles = [
        Line2D([0], [0], marker="o", color=c_bl, linewidth=1.8, label="Baseline (10K warmup)"),
        Line2D([0], [0], marker="s", color=c_hd, linewidth=1.8, label="hd10: Hint $d{=}4$ + 10% drop"),
        Line2D([0], [0], marker="D", color=c_ms, linewidth=1.8, label="ms46: Multi-scale $d{=}4{+}d{=}6$"),
        Line2D([0], [0], marker="^", color=c_msd, linewidth=1.8, label="ms46+drop: MS + 10% drop"),
        Line2D([0], [0], marker="o", color="gray", markersize=10, markeredgecolor="black",
               markeredgewidth=1.5, fillstyle="none", linewidth=0, label="Best checkpoint"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              framealpha=0.9, edgecolor="0.8", fancybox=True)

    # --- Inset: smoothed training loss ---
    loss_json = "/scratch/gpfs/EHAZAN/jh1161/analysis/baseline_vs_best100k/training_loss_data.json"
    if os.path.exists(loss_json):
        with open(loss_json) as f:
            loss_data = json.load(f)

        ax_inset = ax.inset_axes([0.42, 0.58, 0.28, 0.32])  # [x, y, width, height]

        for model_key, color, label in [
            ("baseline", c_bl, "Baseline"),
            ("hd10", c_hd, "hd10"),
        ]:
            steps_raw = np.array(loss_data[model_key]["train/quantile_loss_step"]["steps"])
            vals_raw = np.array(loss_data[model_key]["train/quantile_loss_step"]["values"])

            # Exponential moving average smoothing
            alpha = 0.02
            smoothed = np.zeros_like(vals_raw)
            smoothed[0] = vals_raw[0]
            for i in range(1, len(vals_raw)):
                smoothed[i] = alpha * vals_raw[i] + (1 - alpha) * smoothed[i - 1]

            # Convert steps to thousands
            steps_k = steps_raw / 1000.0
            ax_inset.plot(steps_k, smoothed, color=color, linewidth=1.2, label=label)

        ax_inset.set_xlabel("Steps (K)", fontsize=7)
        ax_inset.set_ylabel("Train Loss", fontsize=7)
        ax_inset.set_title("Training Loss (smoothed)", fontsize=8)
        ax_inset.tick_params(labelsize=6)
        ax_inset.legend(fontsize=6, frameon=False)
        ax_inset.set_xlim(0, 100)
        ax_inset.spines["top"].set_visible(False)
        ax_inset.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save
    pdf_path = os.path.join(FIG_DIR, "convergence_comparison.pdf")
    png_path = os.path.join(FIG_DIR, "convergence_comparison.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # --- Print summary statistics ---
    print("\n--- Convergence Summary ---")
    print(f"Baseline best:        MASE {bl_best_val:.4f} at {bl_best_step}K steps")
    print(f"hd10 best:            MASE {hd_best_val:.4f} at {hd_best_step}K steps "
          f"({(hd_best_val - bl_best_val)/bl_best_val*100:+.2f}%)")
    print(f"ms46 best:            MASE {ms_best_val:.4f} at {ms_best_step}K steps "
          f"({(ms_best_val - bl_best_val)/bl_best_val*100:+.2f}%)")

    print(f"\nConvergence speed: hd10 first beats baseline best ({bl_best_val:.4f}) "
          f"at {convergence_step}K steps with MASE {convergence_val:.4f}")

    # Gap at each shared checkpoint
    print("\n--- Per-Checkpoint Improvement (hd10 vs baseline) ---")
    for s, bm, hm in zip(
        [20, 30, 50, 70, 100],
        baseline_mase[baseline_steps >= 20],
        hd10_mase,
    ):
        delta = (hm - bm) / bm * 100
        print(f"  {s:>3}K: baseline={bm:.4f}, hd10={hm:.4f}, delta={delta:+.2f}%")


if __name__ == "__main__":
    main()
