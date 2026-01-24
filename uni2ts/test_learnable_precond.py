
import torch
import numpy as np
from uni2ts.module.learnable_precondition import LearnablePrecondition

def test_learnable_precondition():
    print("Testing LearnablePrecondition...")
    
    # 1. Test initialization
    module = LearnablePrecondition(degree=2, polynomial_type="chebyshev")
    print(f"Coeffs: {module.coeffs.data}")
    assert module.coeffs.requires_grad
    
    # 2. Test forward (simple case)
    # y_new[t] = y[t] + c1*y[t-1] + c2*y[t-2]
    # coeffs = [c1, c2]
    c1, c2 = module.coeffs.data.tolist()
    
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]) # [1, 4, 1]
    # Expected:
    # t=0: 1.0 (history 0)
    # t=1: 2.0 + c1*1.0
    # t=2: 3.0 + c1*2.0 + c2*1.0
    # t=3: 4.0 + c1*3.0 + c2*2.0
    
    out = module(x)
    print(f"Input: {x.squeeze()}")
    print(f"Output: {out.squeeze()}")
    
    expected_0 = 1.0
    expected_1 = 2.0 + c1*1.0
    expected_2 = 3.0 + c1*2.0 + c2*1.0
    expected_3 = 4.0 + c1*3.0 + c2*2.0
    
    assert torch.allclose(out[0,0,0], torch.tensor(expected_0))
    assert torch.allclose(out[0,1,0], torch.tensor(expected_1))
    assert torch.allclose(out[0,2,0], torch.tensor(expected_2))
    assert torch.allclose(out[0,3,0], torch.tensor(expected_3))
    print("Simple forward test passed.")
    
    # 3. Test masking with sample_id
    # sample_id: [0, 0, 1, 1]
    # t=2 (first element of sample 1) should not use t=1 (sample 0) history
    sample_id = torch.tensor([[0, 0, 1, 1]])
    
    out_masked = module(x, sample_id=sample_id)
    print(f"Output masked: {out_masked.squeeze()}")
    
    # t=0: 1.0
    # t=1: 2.0 + c1*1.0 (same sample)
    # t=2: 3.0 (history from t=1 is masked because sample_id changed)
    # t=3: 4.0 + c1*3.0 (history from t=2 is valid)
    
    expected_m_0 = 1.0
    expected_m_1 = 2.0 + c1*1.0
    expected_m_2 = 3.0 # Reset
    expected_m_3 = 4.0 + c1*3.0
    
    assert torch.allclose(out_masked[0,0,0], torch.tensor(expected_m_0))
    assert torch.allclose(out_masked[0,1,0], torch.tensor(expected_m_1))
    assert torch.allclose(out_masked[0,2,0], torch.tensor(expected_m_2))
    assert torch.allclose(out_masked[0,3,0], torch.tensor(expected_m_3))
    print("Masking test passed.")
    
    # 4. Test reverse
    # Reverse should undo forward
    reversed_x = module.reverse(out)
    print(f"Reversed: {reversed_x.squeeze()}")
    assert torch.allclose(reversed_x, x, atol=1e-5)
    print("Reverse test passed.")
    
    # 5. Test reverse with masking
    reversed_masked = module.reverse(out_masked, sample_id=sample_id)
    print(f"Reversed masked: {reversed_masked.squeeze()}")
    assert torch.allclose(reversed_masked, x, atol=1e-5)
    print("Reverse masked test passed.")

if __name__ == "__main__":
    test_learnable_precondition()
