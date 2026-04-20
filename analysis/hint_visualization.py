"""
Qualitative visualization of polynomial hint signals on time series data.

Creates Figure 2 for the paper: shows raw time series, FIR-filtered output,
and the hint signal (residual) for different polynomial types and degrees.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uni2ts" / "src"))
from uni2ts.common.precondition import compute_polynomial_coefficients

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def apply_fir_filter(signal, coeffs, stride=16):
    """Apply FIR filter: y[t] = x[t] + sum_k c_k * x[t - (k+1)*stride]"""
    T = len(signal)
    filtered = signal.copy()
    for k, c in enumerate(coeffs):
        shift = (k + 1) * stride
        if shift < T:
            filtered[shift:] += c * signal[:T - shift]
    return filtered


def generate_synthetic_series(T=512, seed=42):
    """Generate a synthetic time series with trend, seasonality, and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    # Trend
    trend = 0.01 * t + 0.00002 * t**2
    # Seasonality (period ~64 = 4 patches)
    seasonal = 3.0 * np.sin(2 * np.pi * t / 64) + 1.5 * np.sin(2 * np.pi * t / 32)
    # Noise
    noise = rng.normal(0, 0.5, T)
    return trend + seasonal + noise


def figure2_hint_visualization():
    """4-panel figure showing raw signal, filtered, hint, and multi-scale hints."""
    T = 512  # 32 patches × 16 steps/patch
    patch_size = 16
    signal = generate_synthetic_series(T)

    # Normalize like Moirai2 does (loc-scale)
    loc = np.median(signal)
    scale = np.maximum(np.abs(signal - loc).mean(), 1e-9)
    scaled = (signal - loc) / scale

    # Compute hints for different configs
    coeffs_d4 = compute_polynomial_coefficients("chebyshev", 4)
    coeffs_d6 = compute_polynomial_coefficients("chebyshev", 6)

    filtered_d4 = apply_fir_filter(scaled, coeffs_d4, stride=16)
    filtered_d6 = apply_fir_filter(scaled, coeffs_d6, stride=16)
    hint_d4 = filtered_d4 - scaled
    hint_d6 = filtered_d6 - scaled

    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=150)

    # Panel (a): Raw normalized signal with patch boundaries
    ax = axes[0, 0]
    ax.plot(t, scaled, color='#2c3e50', linewidth=0.8, alpha=0.9)
    for p in range(0, T + 1, patch_size):
        ax.axvline(p, color='gray', alpha=0.15, linewidth=0.5)
    ax.set_title('(a) Normalized input $\\tilde{x}$', fontsize=11)
    ax.set_ylabel('Value')
    ax.set_xlim(0, T)

    # Panel (b): Filtered output vs original
    ax = axes[0, 1]
    ax.plot(t, scaled, color='#bdc3c7', linewidth=0.6, alpha=0.7, label='Input')
    ax.plot(t, filtered_d6, color='#e74c3c', linewidth=0.9, label='FIR output $p(L^S)\\tilde{x}$')
    for p in range(0, T + 1, patch_size):
        ax.axvline(p, color='gray', alpha=0.15, linewidth=0.5)
    ax.set_title('(b) Chebyshev $d{=}6$, stride${{=}}16$ filter output', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, T)

    # Panel (c): Hint signal = filtered - original
    ax = axes[1, 0]
    ax.fill_between(t, 0, hint_d6, where=hint_d6 >= 0, color='#3498db', alpha=0.4)
    ax.fill_between(t, 0, hint_d6, where=hint_d6 < 0, color='#e74c3c', alpha=0.4)
    ax.plot(t, hint_d6, color='#2c3e50', linewidth=0.6)
    ax.axhline(0, color='gray', linewidth=0.5)
    for p in range(0, T + 1, patch_size):
        ax.axvline(p, color='gray', alpha=0.15, linewidth=0.5)
    ax.set_title('(c) Hint signal $h = p(L^S)\\tilde{x} - \\tilde{x}$', fontsize=11)
    ax.set_ylabel('Value')
    ax.set_xlabel('Time step')
    ax.set_xlim(0, T)

    # Panel (d): Multi-scale hints (d=4 and d=6 overlaid)
    ax = axes[1, 1]
    ax.plot(t, hint_d4, color='#e67e22', linewidth=0.8, alpha=0.8, label='$d{=}4$ hint (short-range)')
    ax.plot(t, hint_d6, color='#8e44ad', linewidth=0.8, alpha=0.8, label='$d{=}6$ hint (long-range)')
    ax.axhline(0, color='gray', linewidth=0.5)
    for p in range(0, T + 1, patch_size):
        ax.axvline(p, color='gray', alpha=0.15, linewidth=0.5)
    ax.set_title('(d) Multi-scale hints ($d{=}4 + d{=}6$)', fontsize=11)
    ax.set_xlabel('Time step')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, T)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'hint_visualization.{ext}', bbox_inches='tight')
    plt.close()
    print(f"Saved hint_visualization.pdf/png to {OUT_DIR}")


def figure_hint_patch_structure():
    """Show how hints create inter-patch structure at stride=patch_size."""
    T = 128  # 8 patches
    patch_size = 16
    signal = generate_synthetic_series(T, seed=7)
    loc = np.median(signal)
    scale = np.maximum(np.abs(signal - loc).mean(), 1e-9)
    scaled = (signal - loc) / scale

    coeffs_d6 = compute_polynomial_coefficients("chebyshev", 6)

    # Show stride=1 vs stride=16
    hint_s1 = apply_fir_filter(scaled, coeffs_d6, stride=1) - scaled
    hint_s16 = apply_fir_filter(scaled, coeffs_d6, stride=16) - scaled

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), dpi=150)
    t = np.arange(T)
    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    # Panel (a): Input with patches colored
    ax = axes[0]
    for p in range(T // patch_size):
        start, end = p * patch_size, (p + 1) * patch_size
        ax.fill_between(t[start:end], scaled[start:end],
                        alpha=0.3, color=colors[p])
        ax.plot(t[start:end], scaled[start:end], color=colors[p], linewidth=1.2)
    ax.set_title('(a) Input $\\tilde{x}$ (patches colored)', fontsize=11)
    ax.set_ylabel('Value')
    ax.set_xlim(0, T)

    # Panel (b): Stride=1 hint (dense, within-patch mixing)
    ax = axes[1]
    for p in range(T // patch_size):
        start, end = p * patch_size, (p + 1) * patch_size
        ax.plot(t[start:end], hint_s1[start:end], color=colors[p], linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_title('(b) Stride${{=}}1$ hint (mixes within-patch positions)', fontsize=11)
    ax.set_ylabel('Hint value')
    ax.set_xlim(0, T)

    # Panel (c): Stride=16 hint (patch-aligned, clean inter-patch structure)
    ax = axes[2]
    for p in range(T // patch_size):
        start, end = p * patch_size, (p + 1) * patch_size
        ax.plot(t[start:end], hint_s16[start:end], color=colors[p], linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_title('(c) Stride${{=}}16$ hint (patch-aligned, clean inter-patch)', fontsize=11)
    ax.set_ylabel('Hint value')
    ax.set_xlabel('Time step')
    ax.set_xlim(0, T)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'hint_patch_structure.{ext}', bbox_inches='tight')
    plt.close()
    print(f"Saved hint_patch_structure.pdf/png to {OUT_DIR}")


if __name__ == "__main__":
    figure2_hint_visualization()
    figure_hint_patch_structure()
    print("Done.")
