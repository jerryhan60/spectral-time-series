import torch
import numpy as np
from uni2ts.module.learnable_precondition import LearnablePrecondition

def test_autoregressive_reversal():
    """
    Verify that LearnablePrecondition.reverse correctly inverts the forward pass.
    
    Forward: y'[t] = y[t] + sum(c[i] * y[t-i-1])
    Reverse: y[t] = y'[t] - sum(c[i] * y[t-i-1])
    """
    degree = 2
    # c1 = 0.5, c2 = -0.2
    coeffs = torch.tensor([0.5, -0.2])
    
    module = LearnablePrecondition(degree=degree, polynomial_type="chebyshev")
    # Manually set coefficients
    module.coeffs = torch.nn.Parameter(coeffs)
    
    # Create input sequence
    # batch=1, time=10, dim=1
    y = torch.randn(1, 10, 1)
    
    # Forward pass
    y_prime = module(y)
    
    # Manual forward calculation to verify
    y_manual = y.clone()
    for t in range(10):
        term = 0
        if t >= 1:
            term += coeffs[0] * y[:, t-1, :]
        if t >= 2:
            term += coeffs[1] * y[:, t-2, :]
        y_manual[:, t, :] += term
        
    print(f"Forward check diff: {(y_prime - y_manual).abs().max().item()}")
    assert torch.allclose(y_prime, y_manual, atol=1e-5)
    
    # Reverse pass
    y_recovered = module.reverse(y_prime)
    
    print(f"Reverse check diff: {(y_recovered - y).abs().max().item()}")
    
    # Check first few elements explicitly
    # t=0: y'[0] = y[0] -> y[0] = y'[0]
    # t=1: y'[1] = y[1] + c1*y[0] -> y[1] = y'[1] - c1*y[0]
    # t=2: y'[2] = y[2] + c1*y[1] + c2*y[0] -> y[2] = y'[2] - (c1*y[1] + c2*y[0])
    
    assert torch.allclose(y_recovered, y, atol=1e-5)
    print("Autoregressive reversal test passed!")

if __name__ == "__main__":
    test_autoregressive_reversal()
