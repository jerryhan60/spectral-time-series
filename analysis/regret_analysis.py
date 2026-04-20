"""
Regret bound analysis connecting Hazan & Marsden (2025) theory to empirical results.

Shows:
(a) Monic Chebyshev vs Legendre max norms — why Cheb is minimax optimal
(b) Multi-scale zero distribution — why d=4+d=6 gives complementary bands
(c) Empirical MASE improvement correlates with theoretical prediction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numpy.polynomial import chebyshev, legendre

rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

FIGDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"


def monic_max_norm(poly_type, degree, n_points=10000):
    """Compute max|p(x)| on [-1,1] for monic polynomial of given type and degree."""
    x = np.linspace(-1, 1, n_points)
    if poly_type == 'chebyshev':
        c = [0] * (degree + 1)
        c[degree] = 1
        vals = chebyshev.chebval(x, c)
        # Make monic: divide by 2^{n-1}
        vals = vals / (2 ** (degree - 1))
    elif poly_type == 'legendre':
        c = [0] * (degree + 1)
        c[degree] = 1
        vals = legendre.legval(x, c)
        # Convert to monic
        from numpy.polynomial import legendre as leg_module
        poly = leg_module.Legendre(c)
        power_coeffs = leg_module.leg2poly(poly.coef)
        leading = power_coeffs[-1]
        vals = vals / leading
    return np.max(np.abs(vals))


def get_zeros(poly_type, degree):
    """Get zeros of monic polynomial on [-1,1]."""
    if poly_type == 'chebyshev':
        return np.cos(np.pi * (2 * np.arange(1, degree + 1) - 1) / (2 * degree))
    elif poly_type == 'legendre':
        c = [0] * (degree + 1)
        c[degree] = 1
        roots = legendre.legroots(c)
        return np.sort(roots.real)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    degrees = range(2, 13)

    # (a) Max norm comparison
    ax = axes[0]
    cheb_norms = [monic_max_norm('chebyshev', d) for d in degrees]
    leg_norms = [monic_max_norm('legendre', d) for d in degrees]

    ax.semilogy(list(degrees), cheb_norms, 'o-', color='#1f77b4', linewidth=2,
                markersize=6, label='Chebyshev (minimax)')
    ax.semilogy(list(degrees), leg_norms, 's-', color='#ff7f0e', linewidth=2,
                markersize=6, label='Legendre')

    # Chebyshev theoretical: 2^{1-n}
    d_arr = np.array(list(degrees))
    ax.semilogy(d_arr, 2.0**(1-d_arr), '--', color='#1f77b4', linewidth=1, alpha=0.5,
                label=r'$2^{1-n}$ (Cheb theory)')

    # Ratio annotation
    for d in [4, 6, 8]:
        ratio = leg_norms[d-2] / cheb_norms[d-2]
        ax.annotate(f'{ratio:.1f}x', xy=(d, cheb_norms[d-2]),
                    xytext=(d + 0.3, cheb_norms[d-2] * 0.3),
                    fontsize=7, color='gray',
                    arrowprops=dict(arrowstyle='->', color='gray', linewidth=0.5))

    ax.set_xlabel('Polynomial Degree n')
    ax.set_ylabel(r'$\max_{x \in [-1,1]} |p(x)|$')
    ax.set_title(r'(a) Minimax Norm: $\|p\|_\infty$')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # (b) Zero distribution comparison for d=4 and d=6
    ax = axes[1]

    # Chebyshev zeros
    cheb4_zeros = get_zeros('chebyshev', 4)
    cheb6_zeros = get_zeros('chebyshev', 6)
    leg4_zeros = get_zeros('legendre', 4)
    leg6_zeros = get_zeros('legendre', 6)

    # Plot on number line
    y_pos = {'Cheb d=4': 3, 'Cheb d=6': 2, 'Leg d=4': 1, 'Leg d=6': 0}

    for name, zeros, color, marker in [
        ('Cheb d=4', cheb4_zeros, '#1f77b4', 'o'),
        ('Cheb d=6', cheb6_zeros, '#ff7f0e', 's'),
        ('Leg d=4', leg4_zeros, '#1f77b4', 'D'),
        ('Leg d=6', leg6_zeros, '#ff7f0e', 'D'),
    ]:
        y = y_pos[name]
        filled = 'Cheb' in name
        ax.scatter(zeros, [y] * len(zeros), marker=marker, s=80, color=color,
                   edgecolors='black', linewidth=0.5 if filled else 1.5,
                   facecolors=color if filled else 'none', zorder=3)
        ax.text(-1.15, y, name, va='center', ha='right', fontsize=8)

    # Highlight non-overlapping Chebyshev zeros
    ax.axhline(y=2.5, color='gray', linewidth=0.5, linestyle=':')
    ax.fill_between([-1, 1], 1.7, 3.3, color='#E8F5E9', alpha=0.3, zorder=0)
    ax.text(0, 3.5, 'Non-overlapping (complementary)', ha='center', fontsize=8,
            color='#2ca02c', fontstyle='italic')

    ax.set_xlim(-1.3, 1.1)
    ax.set_ylim(-0.7, 4.0)
    ax.set_xlabel('x')
    ax.set_title('(b) Polynomial Zeros on [-1,1]')
    ax.axhline(y=-0.3, color='black', linewidth=0.5)
    ax.set_yticks([])
    ax.grid(True, alpha=0.15, axis='x')

    # (c) Empirical MASE improvement vs theoretical prediction
    ax = axes[2]

    # Empirical data: single-scale Chebyshev and Legendre results
    data = [
        # (degree, family, MASE, label)
        (4, 'Chebyshev', 1.1944, 'Cheb d=4'),
        (6, 'Chebyshev', 1.1836, 'Cheb d=6'),
        (8, 'Chebyshev', 1.2216, 'Cheb d=8'),
        (4, 'Legendre', 1.2363, 'Leg d=4'),
        (6, 'Legendre', 1.2099, 'Leg d=6'),
        (8, 'Legendre', 1.2225, 'Leg d=8'),
    ]

    baseline = 1.2421
    for degree, family, mase, label in data:
        improvement = (baseline - mase) / baseline * 100
        norm = monic_max_norm(family.lower(), degree)
        color = '#1f77b4' if family == 'Chebyshev' else '#ff7f0e'
        marker = 'o' if family == 'Chebyshev' else 's'
        ax.scatter(norm, improvement, c=color, marker=marker, s=100,
                   edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(label, (norm, improvement), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)

    # Multi-scale points
    ax.scatter(0.03, 6.01, c='#2ca02c', marker='*', s=200,
               edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate('MS Cheb\nd=4+d=6', (0.03, 6.01), textcoords="offset points",
                xytext=(-15, -20), fontsize=7, color='#2ca02c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ca02c'))

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel(r'$\|p\|_\infty$ (smaller = milder filter)')
    ax.set_ylabel('MASE Improvement (%)')
    ax.set_title('(c) Norm vs. Improvement')
    ax.grid(True, alpha=0.2)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label='Chebyshev'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', markersize=8, label='Legendre'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ca02c', markersize=12, label='Multi-scale'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/regret_analysis.pdf')
    plt.savefig(f'{FIGDIR}/regret_analysis.png')
    print("Saved regret_analysis.pdf/png")


if __name__ == '__main__':
    main()
