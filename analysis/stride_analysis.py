#!/usr/bin/env python3
"""
Analysis of WHY stride=16 (patch-aligned) works better than stride=1 (dense FIR).

Key insight: The FIR filter operates on raw (flattened) time steps, but the
transformer sees data in patches of size 16. The stride determines the
relationship between FIR taps and the patch structure.

Stride=16 (patch-aligned):
  - Each tap references the SAME position within the previous patch
  - FIR residual is computed across whole-patch intervals
  - The hint captures inter-patch temporal patterns (coarse structure)
  - Patches internally handle sub-patch variation via self-attention

Stride=1 (dense FIR):
  - Taps reference consecutive raw time steps
  - FIR residual mixes information from different positions within a patch
  - Creates "blurred" cross-patch hints that don't align with patch boundaries
  - The hint captures very fine-grained patterns (mostly sub-patch noise)

Usage:
    python analysis/stride_analysis.py
"""

import numpy as np
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def chebyshev_coeffs(degree):
    """Monic Chebyshev polynomial coefficients (power basis, descending)."""
    from numpy.polynomial.chebyshev import Chebyshev
    T_n = Chebyshev.basis(degree)
    poly = T_n.convert(kind=np.polynomial.Polynomial)
    c = poly.coef[::-1]
    c = c / c[0]
    return c


def fir_response(coeffs, stride, n_freqs=2048):
    """Compute magnitude frequency response |H(e^{jw})| for strided FIR."""
    degree = len(coeffs) - 1
    filter_len = degree * stride + 1
    h = np.zeros(filter_len)
    for i, c in enumerate(coeffs):
        h[i * stride] = c
    # The full polynomial FIR is: y[t] = x[t] + sum_{i=0..n-1} c_i * x[t - (i+1)*stride]
    # Transfer function: H(z) = 1 + sum_{i=0..n-1} c_i * z^{-(i+1)*stride}
    # But the "hint" is H(z) - 1, just the residual part
    freqs = np.linspace(0, np.pi, n_freqs)
    H_residual = np.zeros(n_freqs, dtype=complex)
    for i, c in enumerate(coeffs):
        H_residual += c * np.exp(-1j * freqs * (i + 1) * stride)
    return freqs, np.abs(H_residual)


def main():
    print("=" * 70)
    print("STRIDE ANALYSIS: Why Patch-Aligned Stride Works Better")
    print("=" * 70)

    patch_size = 16

    # ============== Empirical Results ==============
    print("\n--- Empirical MASE Results by Stride ---")
    print(f"{'Stride':>8} {'Relation to Patch':>25} {'d=4 MASE':>10} {'d=6 MASE':>10}")
    print("-" * 55)

    stride_results = [
        (1,  "sub-patch (1 raw step)",     None,     1.2598),
        (4,  "sub-patch (4 raw steps)",    None,     1.2346),
        (8,  "half-patch (8 raw steps)",   1.2382,   1.2204),
        (16, "patch-aligned (16 = patch)", 1.1944,   1.1836),
    ]
    for s, desc, d4, d6 in stride_results:
        d4_str = f"{d4:.4f}" if d4 else "—"
        d6_str = f"{d6:.4f}" if d6 else "—"
        print(f"{s:>8} {desc:>25} {d4_str:>10} {d6_str:>10}")

    print(f"\nBaseline (no hints): 1.2421")
    print(f"\nPatch-aligned (s=16) is consistently {(1.2382-1.1944)/(1.2382)*100:.1f}% better than s=8 for d=4")
    print(f"Patch-aligned (s=16) is consistently {(1.2204-1.1836)/(1.2204)*100:.1f}% better than s=8 for d=6")

    # ============== Theoretical Analysis ==============
    print("\n" + "=" * 70)
    print("THEORETICAL ANALYSIS")
    print("=" * 70)

    print("""
1. PATCH-ALIGNMENT HYPOTHESIS

   Moirai-2 processes time series in patches of size P=16. The patching
   operation creates a sequence of tokens:

     x_patched = [x[0:16], x[16:32], x[32:48], ...]

   Each patch is a P-dimensional vector. The transformer operates on these
   patch tokens via self-attention.

   With stride=S, the FIR hint at position t is:
     hint[t] = sum_{k=1..d} c_k * target[t - k*S]

   After re-patching the hint:
     hint_patch[i] = [hint[i*P], hint[i*P+1], ..., hint[i*P+P-1]]

2. WHY STRIDE=16 (= PATCH SIZE) IS OPTIMAL

   When S=16=P, the hint at position t only references positions that are
   exact patch-widths away:
     hint[t] = sum c_k * target[t - k*16]

   For position j within patch i (i.e., t = 16*i + j):
     hint[16*i + j] = sum c_k * target[16*(i-k) + j]

   Each element j within the hint patch ONLY references element j from
   previous patches. This means:

   - hint_patch[i][j] captures the degree-d polynomial trend of
     element j across patches (inter-patch dynamics)
   - The hint is COHERENT within each patch — all elements share the
     same temporal scale of analysis
   - The hint naturally decomposes into patch-scale dynamics vs.
     within-patch variation

3. WHY STRIDE=1 IS SUBOPTIMAL

   When S=1, hint[t] = sum c_k * target[t - k].
   For position j within patch i:
     hint[16*i + j] = sum c_k * target[16*i + j - k]

   For k > j, this references positions from the PREVIOUS patch at
   different sub-positions. The hint mixes sub-patch positions:

   - hint_patch[i][0] references target at positions -1, -2, -3, -4
     (last elements of previous patch)
   - hint_patch[i][15] references target at positions 14, 13, 12, 11
     (middle of current patch)

   This creates INCOHERENT hint patches where each element captures
   a different temporal relationship. The transformer must then learn
   to disentangle these mixed signals.

4. FREQUENCY DOMAIN PERSPECTIVE

   Stride=S creates a "decimated" FIR filter with taps every S samples.
   The effective sampling rate of the filter is 1/S of the raw rate.

   - Stride=1:  Filter sees full bandwidth [0, pi]. The hint captures
                very high-frequency patterns (up to Nyquist of raw data)
                that are largely handled within patches anyway.

   - Stride=16: Filter's Nyquist is pi/16. The hint captures only
                low-frequency inter-patch patterns, providing genuinely
                new information that the patchwise processing misses.

   In other words, stride=16 creates hints about BETWEEN-patch structure
   (which the transformer needs help with), while stride=1 creates hints
   about WITHIN-patch structure (which the transformer already handles).

5. INFORMATION THEORY PERSPECTIVE

   The mutual information I(hint; future | past_patches) is maximized when:
   - The hint provides information NOT already in the patch representation
   - This happens when the hint captures INTER-patch patterns
   - Stride=P makes the hint orthogonal to within-patch variation
""")

    # ============== Frequency Response Comparison ==============
    if HAS_MPL:
        coeffs_d6 = chebyshev_coeffs(6)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, stride, title in [
            (axes[0], 1, "Stride=1 (dense)"),
            (axes[1], 8, "Stride=8 (half-patch)"),
            (axes[2], 16, "Stride=16 (patch-aligned)"),
        ]:
            freqs, mag = fir_response(coeffs_d6, stride)
            # Normalize frequency to be relative to the raw Nyquist
            ax.plot(freqs / np.pi, mag, linewidth=1.5, color='#C44E52')
            # Mark the patch boundary frequency
            f_patch = np.pi / patch_size  # frequency corresponding to one cycle per patch
            ax.axvline(x=f_patch / np.pi, color='green', linestyle='--', linewidth=1,
                       label=f'Patch freq ($\\pi/{patch_size}$)')
            ax.set_xlabel('Normalized Frequency ($\\times \\pi$)')
            ax.set_ylabel('$|H_{\\mathrm{hint}}(e^{j\\omega})|$')
            ax.set_title(title)
            ax.set_xlim(0, 0.5)  # Focus on low frequencies
            ax.legend(frameon=False, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle('Hint Filter Frequency Response (Chebyshev $d{=}6$)', fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "stride_comparison.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, "stride_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nWrote {FIG_DIR}/stride_comparison.pdf")

        # Also: show what the hint "sees" for a synthetic signal
        np.random.seed(42)
        T = 256  # 16 patches of 16
        t = np.arange(T)
        # Signal: slow trend + medium frequency + fast noise
        signal = 2.0 * np.sin(2 * np.pi * t / 128)  # slow: 1 cycle over 8 patches
        signal += 0.5 * np.sin(2 * np.pi * t / 16)   # patch-frequency
        signal += 0.2 * np.random.randn(T)            # noise

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, signal, 'k-', linewidth=0.8)
        axes[0].set_title('Original Signal (slow trend + patch-freq + noise)')
        axes[0].set_ylabel('Value')
        for i in range(0, T, 16):
            axes[0].axvline(x=i, color='gray', linewidth=0.3, alpha=0.5)

        # Stride=1 hint
        for ax, stride, title, color in [
            (axes[1], 1, 'Hint (stride=1, dense)', '#4C72B0'),
            (axes[2], 16, 'Hint (stride=16, patch-aligned)', '#C44E52'),
        ]:
            hint = np.zeros_like(signal)
            n = len(coeffs_d6)
            for pos in range(n * stride, T):
                for k in range(n):
                    hint[pos] += coeffs_d6[k] * signal[pos - (k + 1) * stride]
            ax.plot(t, hint, color=color, linewidth=0.8)
            ax.set_title(title)
            ax.set_ylabel('Hint Value')
            for i in range(0, T, 16):
                ax.axvline(x=i, color='gray', linewidth=0.3, alpha=0.5)

        axes[2].set_xlabel('Time Step (gray lines = patch boundaries)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "stride_hint_example.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, "stride_hint_example.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Wrote {FIG_DIR}/stride_hint_example.pdf")


if __name__ == "__main__":
    main()
