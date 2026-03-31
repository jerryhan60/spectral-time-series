"""
Spectral theory of polynomial hint preconditioning.

Analyzes why Chebyshev FIR filters are optimal for capturing inter-patch dynamics
in patch-based time series transformers. Creates theoretical figures for the paper.
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


def figure_spectral_gap():
    """
    Show the 'spectral gap' that patching creates: transformer attention operates
    at the patch-token level (freq < 1/P), while the raw data has content at
    higher frequencies. The hint bridges this gap.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
    P = 16  # patch size

    # Panel (a): Raw time series PSD
    ax = axes[0]
    T = 4000
    np.random.seed(42)
    t = np.arange(T)
    # Synthetic signal with multiple frequency components
    signal = (3.0 * np.sin(2 * np.pi * t / 256)   # low freq (period 256 = 16 patches)
             + 2.0 * np.sin(2 * np.pi * t / 64)    # medium freq (period 64 = 4 patches)
             + 1.5 * np.sin(2 * np.pi * t / 16)    # patch freq (period = 1 patch)
             + 1.0 * np.sin(2 * np.pi * t / 4)     # high freq (sub-patch)
             + 0.5 * np.random.randn(T))

    freqs = np.fft.rfftfreq(T, d=1)
    psd = np.abs(np.fft.rfft(signal))**2
    ax.semilogy(freqs[:200], psd[:200], color='#2c3e50', linewidth=0.5, alpha=0.5)
    # Smoothed PSD
    window = 5
    psd_smooth = np.convolve(psd[:200], np.ones(window)/window, mode='same')
    ax.semilogy(freqs[:200], psd_smooth, color='#2c3e50', linewidth=1.5)
    ax.axvline(1/P, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'$1/P = 1/{P}$')
    ax.fill_between([1/P, freqs[199]], ax.get_ylim()[0], ax.get_ylim()[1],
                    alpha=0.1, color='#e74c3c')
    ax.set_xlabel('Frequency (cycles/step)')
    ax.set_ylabel('Power spectral density')
    ax.set_title('(a) Raw time series PSD')
    ax.legend(fontsize=9)

    # Panel (b): What transformer "sees" after patching
    ax = axes[1]
    # Patched signal: average per patch
    N = T // P
    patches = signal[:N*P].reshape(N, P)
    patch_means = patches.mean(axis=1)  # rough approximation
    freqs_patch = np.fft.rfftfreq(N, d=1)
    psd_patch = np.abs(np.fft.rfft(patch_means))**2
    ax.semilogy(freqs_patch, psd_patch, color='#3498db', linewidth=1.5)
    ax.set_xlabel('Frequency (cycles/patch)')
    ax.set_ylabel('Power spectral density')
    ax.set_title('(b) Patch-level PSD (attention sees this)')

    # Panel (c): Hint frequency response bridges the gap
    ax = axes[2]
    coeffs_d4 = compute_polynomial_coefficients("chebyshev", 4)
    coeffs_d6 = compute_polynomial_coefficients("chebyshev", 6)

    # FIR response at stride=16 (in raw time domain)
    N_freq = 1024
    omega = np.linspace(0, np.pi, N_freq)
    # At stride S, the FIR filter operates on patch indices:
    # H(z) = 1 + c1*z^{-1} + c2*z^{-2} + ... (in the z^{-S} variable)
    # So in raw frequency: H(omega) = 1 + sum c_k * exp(-j*k*S*omega)
    for label, coeffs, color in [
        ('Chebyshev $d{=}4$', coeffs_d4, '#e67e22'),
        ('Chebyshev $d{=}6$', coeffs_d6, '#8e44ad'),
    ]:
        S = 16
        H = np.ones(N_freq, dtype=complex)
        for k, c in enumerate(coeffs):
            H += c * np.exp(-1j * (k+1) * S * omega)
        ax.plot(omega / np.pi, 20 * np.log10(np.abs(H) + 1e-12),
                color=color, linewidth=1.5, label=label)

    # Mark the patch frequency
    ax.axvline(1/P * 2, color='#e74c3c', linestyle='--', linewidth=1,
               label=f'$2\\pi/P$')
    ax.set_xlabel('Normalized frequency ($\\omega/\\pi$)')
    ax.set_ylabel('|H($\\omega$)| (dB)')
    ax.set_title('(c) Hint filter response (stride=16)')
    ax.legend(fontsize=8)
    ax.set_ylim(-40, 10)
    ax.axhline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'spectral_gap.{ext}', bbox_inches='tight')
    plt.close()
    print(f"Saved spectral_gap.pdf/png")


def figure_minimax_comparison():
    """
    Compare Chebyshev vs Legendre minimax properties on [-1,1].
    Chebyshev has equioscillation = minimax optimal = best spectral spread.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    x = np.linspace(-1, 1, 1000)

    for d, ax in [(4, axes[0]), (6, axes[1])]:
        cheb_coeffs = compute_polynomial_coefficients("chebyshev", d)
        leg_coeffs = compute_polynomial_coefficients("legendre", d)

        # Evaluate monic polynomial: x^d + c1*x^{d-1} + ... + cd
        def eval_monic_poly(coeffs, x, degree):
            y = x**degree
            for k, c in enumerate(coeffs):
                y += c * x**(degree - k - 1)
            return y

        y_cheb = eval_monic_poly(cheb_coeffs, x, d)
        y_leg = eval_monic_poly(leg_coeffs, x, d)

        ax.plot(x, y_cheb, color='#e74c3c', linewidth=1.5,
                label=f'Chebyshev (max={np.max(np.abs(y_cheb)):.4f})')
        ax.plot(x, y_leg, color='#3498db', linewidth=1.5,
                label=f'Legendre (max={np.max(np.abs(y_leg)):.4f})')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axhline(np.max(np.abs(y_cheb)), color='#e74c3c', linestyle=':', alpha=0.5)
        ax.axhline(-np.max(np.abs(y_cheb)), color='#e74c3c', linestyle=':', alpha=0.5)

        # Mark Chebyshev equioscillation points
        from numpy.polynomial import chebyshev, polynomial
        cheb_basis = chebyshev.Chebyshev.basis(d)
        roots = cheb_basis.roots()
        # Chebyshev extrema at cos(k*pi/d) for k=0,...,d
        extrema = np.cos(np.arange(d + 1) * np.pi / d)
        ax.scatter(extrema, eval_monic_poly(cheb_coeffs, extrema, d),
                  color='#e74c3c', s=30, zorder=5)

        ax.set_xlabel('$x$')
        ax.set_ylabel(f'$p_{d}(x)$')
        ax.set_title(f'Monic degree-{d} polynomial on [-1,1]')
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'minimax_comparison.{ext}', bbox_inches='tight')
    plt.close()
    print(f"Saved minimax_comparison.pdf/png")


def figure_hint_spectral_content():
    """
    Show what spectral content the hint adds vs what the raw signal already has.
    The hint = p(L^S)x - x captures frequencies that the polynomial filter passes.
    """
    T = 4000
    P = 16
    np.random.seed(42)
    t = np.arange(T)

    # Signal with known spectral content
    signal = (3 * np.sin(2*np.pi*t/256) + 2 * np.sin(2*np.pi*t/64) +
              1.5 * np.sin(2*np.pi*t/16) + np.sin(2*np.pi*t/4) +
              0.5 * np.random.randn(T))

    # Scale
    loc = np.median(signal)
    scale = max(np.abs(signal - loc).mean(), 1e-9)
    scaled = (signal - loc) / scale

    coeffs_d4 = compute_polynomial_coefficients("chebyshev", 4)
    coeffs_d6 = compute_polynomial_coefficients("chebyshev", 6)

    def apply_fir(s, coeffs, stride):
        out = s.copy()
        for k, c in enumerate(coeffs):
            shift = (k + 1) * stride
            if shift < len(s):
                out[shift:] += c * s[:len(s) - shift]
        return out

    hint_d4 = apply_fir(scaled, coeffs_d4, 16) - scaled
    hint_d6 = apply_fir(scaled, coeffs_d6, 16) - scaled

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=150, sharex=True)
    freqs = np.fft.rfftfreq(T, d=1)

    for ax, sig, label, color in [
        (axes[0], scaled, 'Input $\\tilde{x}$', '#2c3e50'),
        (axes[1], hint_d4, 'Hint $h_{d=4}$', '#e67e22'),
        (axes[2], hint_d6, 'Hint $h_{d=6}$', '#8e44ad'),
    ]:
        psd = np.abs(np.fft.rfft(sig))**2
        window = 5
        psd_smooth = np.convolve(psd[:500], np.ones(window)/window, mode='same')
        ax.semilogy(freqs[:500], psd_smooth, color=color, linewidth=1.2)
        ax.axvline(1/P, color='red', linestyle='--', alpha=0.5, linewidth=1,
                   label='$1/P$' if ax == axes[0] else '')
        ax.set_ylabel('PSD')
        ax.set_title(label, fontsize=11)
        if ax == axes[0]:
            ax.legend(fontsize=9)

    axes[2].set_xlabel('Frequency (cycles/step)')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'hint_spectral_content.{ext}', bbox_inches='tight')
    plt.close()
    print(f"Saved hint_spectral_content.pdf/png")


if __name__ == "__main__":
    figure_spectral_gap()
    figure_minimax_comparison()
    figure_hint_spectral_content()
    print("Done.")
