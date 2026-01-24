import torch
import numpy as np

from uni2ts.transform.patch_precondition import PatchPolynomialPrecondition, PatchReversePrecondition

def test_patch_preconditioning_correctness():
    """
    Test that patch preconditioning applies convolution correctly along time dimension.
    """
    # Setup
    patch_size = 2
    time_steps = 5
    
    # Create synthetic data: (1, time, patch)
    # time 0: [1, 2]
    # time 1: [3, 4]
    # time 2: [5, 6]
    # ...
    data = np.zeros((1, time_steps, patch_size))
    for t in range(time_steps):
        data[0, t, :] = [t*2 + 1, t*2 + 2]
        
    # data:
    # t=0: [1, 2]
    # t=1: [3, 4]
    # t=2: [5, 6]
    # t=3: [7, 8]
    # t=4: [9, 10]
    
    # Coefficients: [c1] = [-0.5] (degree 1 for simplicity)
    # y'_t = y_t + c1 * y_{t-1}
    # y'_t = y_t - 0.5 * y_{t-1}
    
    coeffs = np.array([-0.5])
    
    transform = PatchPolynomialPrecondition(
        degree=1,
        target_field="target",
        enabled=True,
        store_original=True
    )
    # Manually set coeffs for testing
    transform.coeffs = coeffs
    
    data_entry = {"target": data.copy()}
    result_entry = transform(data_entry)
    result = result_entry["target"]
    
    # Expected result:
    # t=0: y_0 (no history) -> [1, 2]
    # t=1: y_1 - 0.5 * y_0 -> [3, 4] - 0.5*[1, 2] = [3-0.5, 4-1] = [2.5, 3.0]
    # t=2: y_2 - 0.5 * y_1 -> [5, 6] - 0.5*[3, 4] = [5-1.5, 6-2] = [3.5, 4.0]
    # ...
    
    expected = np.zeros_like(data)
    expected[0, 0] = data[0, 0]
    for t in range(1, time_steps):
        expected[0, t] = data[0, t] + coeffs[0] * data[0, t-1]
        
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Forward pass correct")
    
    # Test Reversal
    reverse_transform = PatchReversePrecondition(
        target_field="target",
        enabled=True
    )
    
    # We need to pass metadata
    reverse_entry = {
        "target": result.copy(),
        "precondition_coeffs": coeffs,
        "precondition_enabled": True,
        "precondition_is_patched": True
    }
    
    restored_entry = reverse_transform(reverse_entry)
    restored = restored_entry["target"]
    
    np.testing.assert_allclose(restored, data, rtol=1e-5)
    print("Reverse pass correct")

def test_patch_preconditioning_with_context():
    """
    Test reversal with context.
    """
    patch_size = 2
    
    # Context: 2 steps
    context = np.array([
        [[1.0, 2.0], [3.0, 4.0]]
    ]) # (1, 2, 2)
    
    # Sequence to predict: 2 steps
    # t=2 (relative to context start): [5, 6]
    # t=3: [7, 8]
    
    # Original future
    future = np.array([
        [[5.0, 6.0], [7.0, 8.0]]
    ])
    
    coeffs = np.array([-0.5])
    
    # Precondition future manually using context
    # y'_2 = y_2 - 0.5 * y_1 (last of context)
    # y'_3 = y_3 - 0.5 * y_2
    
    precond_future = np.zeros_like(future)
    # t=0 of future corresponds to t=2 overall
    # y_1 is context[0, 1] = [3, 4]
    precond_future[0, 0] = future[0, 0] + coeffs[0] * context[0, 1]
    # t=1 of future corresponds to t=3 overall
    # y_2 is future[0, 0] = [5, 6]
    precond_future[0, 1] = future[0, 1] + coeffs[0] * future[0, 0]
    
    # Now reverse using context
    reverse_transform = PatchReversePrecondition(
        target_field="target",
        enabled=True
    )
    
    reverse_entry = {
        "target": precond_future.copy(),
        "precondition_coeffs": coeffs,
        "precondition_enabled": True,
        "precondition_is_patched": True
    }
    
    restored_entry = reverse_transform(reverse_entry, context=context)
    restored = restored_entry["target"]
    
    np.testing.assert_allclose(restored, future, rtol=1e-5)
    print("Reverse with context correct")

if __name__ == "__main__":
    test_patch_preconditioning_correctness()
    test_patch_preconditioning_with_context()
