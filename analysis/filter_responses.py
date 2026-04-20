#!/usr/bin/env python3
"""
Visualize FIR filter frequency responses for different polynomial families.
Shows why Chebyshev provides better spectral coverage than Legendre.

Usage:
    python analysis/filter_responses.py
"""

import numpy as np
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib required")
    exit(1)

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def chebyshev_coeffs(degree):
    """Monic Chebyshev polynomial coefficients (power basis)."""
    coeffs = np.zeros(degree + 1)
    coeffs[degree] = 1
    if degree == 0:
        return np.array([1.0])
    # Chebyshev recurrence: T_0=1, T_1=x, T_{n+1}=2xT_n - T_{n-1}
    # Then monic = T_n / 2^{n-1}
    T = [np.array([1.0]), np.array([1.0, 0.0])]
    for n in range(2, degree + 1):
        Tn = np.zeros(n + 1)
        # 2x * T_{n-1}
        Tn[:n] += 2 * T[-1]
        # - T_{n-2}
        Tn[2:] -= T[-2]
        # Fix: polynomial mult by x shifts coeffs
        shifted = np.zeros(n + 1)
        shifted[:n] = T[-1]  # x * T_{n-1}
        Tn = np.zeros(n + 1)
        for i in range(n):
            Tn[i] += 2 * T[-1][i]  # 2x * T_{n-1}[i] -> shift
        # Actually let's use numpy
        T.append(Tn)
    # Use numpy for correctness
    from numpy.polynomial.chebyshev import Chebyshev
    T_n = Chebyshev.basis(degree)
    # Convert to power basis
    poly = T_n.convert(kind=np.polynomial.Polynomial)
    c = poly.coef[::-1]  # highest degree first
    # Make monic
    c = c / c[0]
    return c


def legendre_coeffs(degree):
    """Monic Legendre polynomial coefficients (power basis)."""
    from numpy.polynomial.legendre import Legendre
    P_n = Legendre.basis(degree)
    poly = P_n.convert(kind=np.polynomial.Polynomial)
    c = poly.coef[::-1]
    c = c / c[0]
    return c


def fir_response(coeffs, stride, n_freqs=1024):
    """Compute magnitude frequency response of FIR filter with given coefficients and stride."""
    # The FIR filter has taps at positions 0, stride, 2*stride, ..., degree*stride
    degree = len(coeffs) - 1
    filter_len = degree * stride + 1
    h = np.zeros(filter_len)
    for i, c in enumerate(coeffs):
        h[i * stride] = c
    freqs = np.linspace(0, np.pi, n_freqs)
    H = np.zeros(n_freqs, dtype=complex)
    for k, w in enumerate(freqs):
        for n in range(filter_len):
            H[k] += h[n] * np.exp(-1j * w * n)
    return freqs, np.abs(H)


def main():
    stride = 16

    # ========== Figure: Frequency responses for different families at d=6 ==========
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    families = [
        ("Chebyshev", chebyshev_coeffs),
        ("Legendre", legendre_coeffs),
    ]

    # (a) Single-scale d=6 comparison
    ax = axes[0, 0]
    for name, coeff_fn in families:
        c = coeff_fn(6)
        freqs, mag = fir_response(c, stride)
        ax.plot(freqs / np.pi, 20 * np.log10(mag + 1e-10), label=name, linewidth=1.5)
    ax.set_xlabel('Normalized Frequency ($\\times \\pi$)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('(a) $d{=}6$, stride=16')
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Single-scale d=4 comparison
    ax = axes[0, 1]
    for name, coeff_fn in families:
        c = coeff_fn(4)
        freqs, mag = fir_response(c, stride)
        ax.plot(freqs / np.pi, 20 * np.log10(mag + 1e-10), label=name, linewidth=1.5)
    ax.set_xlabel('Normalized Frequency ($\\times \\pi$)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('(b) $d{=}4$, stride=16')
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) Multi-scale: Chebyshev d=4 + d=6 (overlay)
    ax = axes[1, 0]
    c4 = chebyshev_coeffs(4)
    c6 = chebyshev_coeffs(6)
    f4, m4 = fir_response(c4, stride)
    f6, m6 = fir_response(c6, stride)
    ax.plot(f4 / np.pi, 20 * np.log10(m4 + 1e-10), label='Cheb $d{=}4$', linewidth=1.5, color='#4C72B0')
    ax.plot(f6 / np.pi, 20 * np.log10(m6 + 1e-10), label='Cheb $d{=}6$', linewidth=1.5, color='#C44E52')
    ax.set_xlabel('Normalized Frequency ($\\times \\pi$)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('(c) Multi-scale Chebyshev')
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (d) Multi-scale: Legendre d=4 + d=6 (overlay)
    ax = axes[1, 1]
    l4 = legendre_coeffs(4)
    l6 = legendre_coeffs(6)
    f4, m4 = fir_response(l4, stride)
    f6, m6 = fir_response(l6, stride)
    ax.plot(f4 / np.pi, 20 * np.log10(m4 + 1e-10), label='Leg $d{=}4$', linewidth=1.5, color='#8172B2')
    ax.plot(f6 / np.pi, 20 * np.log10(m6 + 1e-10), label='Leg $d{=}6$', linewidth=1.5, color='#CCB974')
    ax.set_xlabel('Normalized Frequency ($\\times \\pi$)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('(d) Multi-scale Legendre')
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('FIR Filter Frequency Responses by Polynomial Family', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "filter_responses.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "filter_responses.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wrote {FIG_DIR}/filter_responses.pdf")

    # ========== Figure: Zero locations ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    for ax, (name, coeff_fn), color in [(ax1, families[0], '#C44E52'), (ax2, families[1], '#8172B2')]:
        for d in [4, 6]:
            c = coeff_fn(d)
            roots = np.roots(c)
            real_roots = roots[np.abs(roots.imag) < 1e-10].real
            ax.scatter(real_roots, [d] * len(real_roots), s=60, alpha=0.7,
                       color=color, edgecolors='black', linewidth=0.5, zorder=3)
        ax.set_xlabel('Zero Location')
        ax.set_ylabel('Degree')
        ax.set_title(f'{name} Zeros')
        ax.set_yticks([4, 6])
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Polynomial Zero Distributions', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "zero_locations.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "zero_locations.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wrote {FIG_DIR}/zero_locations.pdf")

    # Print coefficient info
    print("\n--- Filter Coefficients ---")
    for name, coeff_fn in families:
        for d in [4, 6]:
            c = coeff_fn(d)
            print(f"{name} d={d}: coeffs={np.round(c, 4)}, max|c|={np.max(np.abs(c)):.4f}")
            roots = np.roots(c)
            real_roots = roots[np.abs(roots.imag) < 1e-10].real
            print(f"  real zeros: {np.sort(real_roots).round(4)}")


if __name__ == "__main__":
    main()
