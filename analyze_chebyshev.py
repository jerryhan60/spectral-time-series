
import numpy as np
from numpy.polynomial import chebyshev, polynomial
import matplotlib.pyplot as plt

def get_coeffs(n):
    # Replicating logic from precondition.py
    cheb = chebyshev.Chebyshev.basis(n)
    power_poly = cheb.convert(kind=polynomial.Polynomial)
    power_coeffs = power_poly.coef
    leading_coeff = power_coeffs[-1]
    monic_coeffs = power_coeffs / leading_coeff
    # [c1, c2, ..., cn]
    return monic_coeffs[:-1][::-1]

print("Degree | Coefficients (c1, c2, ...)")
print("-" * 40)

responses = {}
degrees = [1, 2, 3, 4, 5]

for d in degrees:
    coeffs = get_coeffs(d)
    print(f"d={d}  | {coeffs}")
    
    # Frequency response H(z) = 1 + c1 z^-1 + ... + cn z^-n
    # Evaluate on unit circle z = exp(j * omega)
    w = np.linspace(0, np.pi, 1000) # 0 to Nyquist
    z = np.exp(1j * w)
    
    # H(z) = 1 + sum(c_i * z^-i)
    H = np.ones_like(z)
    for i, c in enumerate(coeffs):
        H += c * z**(-(i+1))
        
    responses[d] = np.abs(H)

    # Check for max amplification
    max_amp = np.max(np.abs(H))
    freq_of_max = w[np.argmax(np.abs(H))] / np.pi
    print(f"      Max Amp: {max_amp:.2f} @ {freq_of_max:.2f}*pi")

# Simple text-based plot of response at pi (Nyquist)
print("\nResponse at Nyquist (High Freq Noise):")
for d in degrees:
    resp_pi = responses[d][-1]
    print(f"d={d}: {resp_pi:.4f}")
