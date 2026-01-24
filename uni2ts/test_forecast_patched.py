import torch
import numpy as np
from uni2ts.model.moirai.forecast_patched import MoiraiForecastPatched
from uni2ts.distribution import StudentTOutput

def test_forecast_patched_init():
    """
    Test that MoiraiForecastPatched initializes correctly.
    """
    module_kwargs = {
        "distr_output": StudentTOutput(),
        "d_model": 128,
        "num_layers": 2,
        "patch_sizes": [32, 64],
        "max_seq_len": 32,
        "attn_dropout_p": 0.0,
        "dropout_p": 0.0,
    }
    
    model = MoiraiForecastPatched(
        prediction_length=10,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=64,
        module_kwargs=module_kwargs,
        patch_size="auto",
        num_samples=10,
        enable_preconditioning=True,
        precondition_type="chebyshev",
        precondition_degree=2,
        reverse_output=True,
    )
    
    print("Forecast model initialized successfully")
    return model

def test_forecast_patched_convert():
    """
    Test _convert method with preconditioning.
    """
    model = test_forecast_patched_init()
    
    patch_size = 32
    batch_size = 2
    past_time = 64
    tgt_dim = 1
    
    # Create dummy data
    past_target = torch.randn(batch_size, past_time, tgt_dim)
    past_observed_target = torch.ones(batch_size, past_time, tgt_dim, dtype=torch.bool)
    past_is_pad = torch.zeros(batch_size, past_time, dtype=torch.bool)
    
    # Run _convert
    (
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
    ) = model._convert(
        patch_size,
        past_target,
        past_observed_target,
        past_is_pad,
    )
    
    print("Convert ran successfully")
    
    # Check if target is modified (preconditioned)
    # Since we use random data, it's hard to check exact values without reimplementing logic.
    # But we can check if it runs without error.
    
    max_patch_size = max(model.module.patch_sizes)
    assert target.shape[-1] == max_patch_size
    
    # Check if we can reverse
    # We need to mock samples
    # samples shape: (num_samples, batch, combine_seq, patch)
    combine_seq = target.shape[1]
    samples = torch.randn(5, batch_size, combine_seq, patch_size)
    
    reversed_samples = model._reverse_samples(samples, variate_id)
    
    print("Reverse samples ran successfully")
    assert reversed_samples.shape == samples.shape

if __name__ == "__main__":
    test_forecast_patched_convert()
