"""
Spectral analysis of FIR hint filters.

Creates publication-quality figure showing:
1. Frequency response of Chebyshev FIR filters at various degrees
2. Comparison of Chebyshev vs Legendre vs L2-optimal frequency responses
3. Multi-scale complementarity (d=4 and d=6 cover different bands)
4. Patch-aligned stride effect on spectral coverage
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numpy.polynomial import chebyshev, legendre

# Publication settings
rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

def get_chebyshev_coeffs(degree, stride):
    """Get Chebyshev FIR filter coefficients (monic polynomial)."""
    coeffs = np.zeros(degree + 1)
    coeffs[degree] = 1  # basis polynomial T_n
    # Convert to power basis
    power_coeffs = chebyshev.cheb2poly(coeffs)
    # Evaluate at stride-spaced points to get FIR taps
    n_taps = degree + 1
    return power_coeffs

def get_legendre_coeffs(degree, stride):
    """Get Legendre FIR filter coefficients (monic polynomial)."""
    coeffs = np.zeros(degree + 1)
    coeffs[degree] = 1
    power_coeffs = legendre.leg2poly(coeffs)
    return power_coeffs

def freq_response(coeffs, stride, n_freq=2048):
    """Compute frequency response of an FIR filter with given coefficients and stride."""
    n_taps = len(coeffs)
    freqs = np.linspace(0, np.pi, n_freq)
    H = np.zeros(n_freq, dtype=complex)
    for k, c in enumerate(coeffs):
        H += c * np.exp(-1j * freqs * k * stride)
    return freqs, np.abs(H)

def freq_response_residual(coeffs, stride, n_freq=2048):
    """Compute frequency response of (I - FIR) = residual hint."""
    freqs = np.linspace(0, np.pi, n_freq)
    n_taps = len(coeffs)
    # The hint is precond_target - target = (FIR * target) - target
    # In frequency domain: H_hint(w) = H_fir(w) - 1
    H_fir = np.zeros(n_freq, dtype=complex)
    for k, c in enumerate(coeffs):
        H_fir += c * np.exp(-1j * freqs * k * stride)
    H_hint = H_fir - 1.0
    return freqs, np.abs(H_hint)

# --- Figure 1: Multi-scale complementarity ---
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

# Panel A: Individual Chebyshev frequency responses
ax = axes[0, 0]
stride = 16
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, d in enumerate([2, 4, 6, 8]):
    coeffs = get_chebyshev_coeffs(d, stride)
    freqs, mag = freq_response_residual(coeffs, stride)
    # Normalize frequency to [0, 1] where 1 = Nyquist
    ax.plot(freqs / np.pi, mag / mag.max(), color=colors[i], label=f'd={d}', linewidth=1.5)

# Mark patch frequency
ax.axvline(x=1/stride * 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.text(1/stride * 2 + 0.02, 0.95, 'f = 1/P', fontsize=7, color='gray', va='top')
ax.set_xlabel('Normalized frequency (f/f_Nyquist)')
ax.set_ylabel('Normalized magnitude')
ax.set_title('(a) Chebyshev FIR residual responses')
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Panel B: d=4 + d=6 multi-scale complementarity
ax = axes[0, 1]
coeffs4 = get_chebyshev_coeffs(4, stride)
coeffs6 = get_chebyshev_coeffs(6, stride)
freqs, mag4 = freq_response_residual(coeffs4, stride)
freqs, mag6 = freq_response_residual(coeffs6, stride)
# Combined coverage = max of individual responses
mag_combined = np.maximum(mag4, mag6)

ax.fill_between(freqs / np.pi, 0, mag4 / mag_combined.max(), alpha=0.3, color='#1f77b4', label='d=4 band')
ax.fill_between(freqs / np.pi, 0, mag6 / mag_combined.max(), alpha=0.3, color='#ff7f0e', label='d=6 band')
ax.plot(freqs / np.pi, mag_combined / mag_combined.max(), 'k-', linewidth=1.5, label='Combined')
ax.axvline(x=1/stride * 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_xlabel('Normalized frequency (f/f_Nyquist)')
ax.set_ylabel('Normalized magnitude')
ax.set_title('(b) Multi-scale d=4+d=6 coverage')
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Panel C: Chebyshev vs Legendre comparison
ax = axes[1, 0]
for d in [4, 6]:
    cheb_coeffs = get_chebyshev_coeffs(d, stride)
    leg_coeffs = get_legendre_coeffs(d, stride)
    freqs, cheb_mag = freq_response_residual(cheb_coeffs, stride)
    freqs, leg_mag = freq_response_residual(leg_coeffs, stride)

    color = '#1f77b4' if d == 4 else '#ff7f0e'
    ax.plot(freqs / np.pi, cheb_mag / cheb_mag.max(), color=color, linewidth=1.5,
            label=f'Cheb d={d}')
    ax.plot(freqs / np.pi, leg_mag / leg_mag.max(), color=color, linewidth=1.5,
            linestyle='--', label=f'Leg d={d}')

ax.axvline(x=1/stride * 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_xlabel('Normalized frequency (f/f_Nyquist)')
ax.set_ylabel('Normalized magnitude')
ax.set_title('(c) Chebyshev vs Legendre')
ax.legend(loc='upper right', framealpha=0.8, ncol=2)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Panel D: Stride effect
ax = axes[1, 1]
d = 6
strides = [1, 4, 8, 16]
colors_stride = ['#d62728', '#9467bd', '#8c564b', '#2ca02c']
for i, s in enumerate(strides):
    coeffs = get_chebyshev_coeffs(d, s)
    freqs, mag = freq_response_residual(coeffs, s)
    ax.plot(freqs / np.pi, mag / mag.max(), color=colors_stride[i], linewidth=1.5,
            label=f'stride={s}')

# Mark patch frequency for stride=16
ax.axvline(x=1/16 * 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.text(1/16 * 2 + 0.02, 0.95, 'f = 1/P', fontsize=7, color='gray', va='top')
ax.set_xlabel('Normalized frequency (f/f_Nyquist)')
ax.set_ylabel('Normalized magnitude')
ax.set_title('(d) Effect of stride (d=6)')
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/scratch/gpfs/EHAZAN/jh1161/analysis/figures/spectral_analysis.pdf')
plt.savefig('/scratch/gpfs/EHAZAN/jh1161/analysis/figures/spectral_analysis.png')
print("Saved spectral_analysis.pdf/png")

# --- Figure 2: Zero distribution of polynomial filters ---
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# Panel A: Chebyshev zeros on unit circle
ax = axes[0]
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5, alpha=0.3)

for d, color in [(4, '#1f77b4'), (6, '#ff7f0e'), (8, '#2ca02c')]:
    # Chebyshev zeros: cos((2k-1)π/(2n)) for k=1..n
    zeros_angle = [(2*k - 1) * np.pi / (2*d) for k in range(1, d+1)]
    zeros_x = np.cos(zeros_angle)
    zeros_y = np.sin(zeros_angle)
    ax.scatter(zeros_x, zeros_y, s=40, color=color, label=f'd={d}', zorder=5, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_title('(a) Chebyshev zeros (equioscillation)')
ax.set_aspect('equal')
ax.legend(loc='lower left', framealpha=0.8)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-0.1, 1.3)
ax.grid(True, alpha=0.2)

# Panel B: Comparison of zero distributions
ax = axes[1]
ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5, alpha=0.3)

d = 6
# Chebyshev zeros
zeros_angle_cheb = [(2*k - 1) * np.pi / (2*d) for k in range(1, d+1)]
ax.scatter(np.cos(zeros_angle_cheb), np.sin(zeros_angle_cheb), s=50, color='#1f77b4',
           marker='o', label='Chebyshev', zorder=5, edgecolors='black', linewidth=0.5)

# Legendre zeros (roots of P_6)
leg_coeffs = np.zeros(d+1)
leg_coeffs[d] = 1
leg_roots = legendre.legroots(leg_coeffs)
# Map to unit circle angles
leg_angles = np.arccos(np.clip(leg_roots.real, -1, 1))
ax.scatter(np.cos(leg_angles), np.sin(leg_angles), s=50, color='#ff7f0e',
           marker='s', label='Legendre', zorder=5, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_title('(b) Chebyshev vs Legendre zeros (d=6)')
ax.set_aspect('equal')
ax.legend(loc='lower left', framealpha=0.8)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-0.1, 1.3)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/scratch/gpfs/EHAZAN/jh1161/analysis/figures/zero_distribution.pdf')
plt.savefig('/scratch/gpfs/EHAZAN/jh1161/analysis/figures/zero_distribution.png')
print("Saved zero_distribution.pdf/png")

# --- Figure 3: Spectral gap illustration ---
fig, ax = plt.subplots(1, 1, figsize=(6, 3))

# Simulate a time series with multiple frequency components
np.random.seed(42)
T = 256
t = np.arange(T)
P = 16  # patch size

# Signal with low-freq + high-freq components
low_freq = np.sin(2 * np.pi * t / 64)  # well below 1/P
mid_freq = 0.5 * np.sin(2 * np.pi * t / 16)  # at 1/P boundary
high_freq = 0.3 * np.sin(2 * np.pi * t / 4)  # well above 1/P

signal = low_freq + mid_freq + high_freq

# Power spectrum
freqs = np.fft.rfftfreq(T)
power = np.abs(np.fft.rfft(signal))**2
power_db = 10 * np.log10(power / power.max() + 1e-10)

ax.plot(freqs, power_db, 'k-', linewidth=1, alpha=0.7)

# Mark patch frequency
patch_freq = 1 / P
ax.axvline(x=patch_freq, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax.text(patch_freq + 0.005, -2, 'f = 1/P', fontsize=9, color='red', va='top')

# Shade the blind spot region
ax.axvspan(patch_freq, 0.5, alpha=0.15, color='red', label='Patching blind spot')
ax.axvspan(0, patch_freq, alpha=0.1, color='green', label='Captured by patches')

# Show where hints help
coeffs = get_chebyshev_coeffs(6, P)
hint_freqs, hint_mag = freq_response_residual(coeffs, P)
hint_norm = hint_mag / hint_mag.max() * (-5)  # Scale for visibility
ax.plot(hint_freqs / (2*np.pi), hint_norm - 25, color='#1f77b4', linewidth=1.5, alpha=0.8)
ax.text(0.15, -33, 'Hint FIR response', fontsize=8, color='#1f77b4')

ax.set_xlabel('Frequency (cycles/sample)')
ax.set_ylabel('Power (dB)')
ax.set_title('Spectral gap from patching and FIR hint coverage')
ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
ax.set_xlim(0, 0.35)
ax.set_ylim(-45, 5)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/scratch/gpfs/EHAZAN/jh1161/analysis/figures/spectral_gap.pdf')
plt.savefig('/scratch/gpfs/EHAZAN/jh1161/analysis/figures/spectral_gap.png')
print("Saved spectral_gap.pdf/png")

print("\nAll spectral analysis figures generated successfully.")
