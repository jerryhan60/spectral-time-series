# Hint Ablation Experiment Design

## Motivation

The best-performing Moirai2 variant uses fixed Chebyshev polynomial FIR filters as extra input channels ("hints"). These ablations isolate what drives the improvement:
- Extra input capacity (more parameters in `in_proj`)?
- Any learned temporal filter?
- The specific Chebyshev polynomial structure?

## Experiment Matrix

| Name | Extra channel | Kernel | Init | Purpose |
|------|--------------|--------|------|---------|
| `ablation_duplicate` | `target` copy | N/A | N/A | Controls for extra capacity |
| `ablation_learned_4tap` | `learned_conv_4(target) - target` | 4 | Chebyshev d=4 | Learned filter, matched receptive field |
| `ablation_learned_16tap` | `learned_conv_16(target) - target` | 16 | Chebyshev d=4 (zero-padded) | Learned filter, full-patch receptive field |
| *(existing)* hint d=4 | `cheb_d4(target) - target` | 4 | Fixed Chebyshev | Reference |

All runs: 10K steps, anomaly_zscore=8.0, 1K warmup, single extra channel, GIFT-Eval benchmark.

## Implementation

### Code changes (module.py only)

Extend `hint_ablation` parameter with two new modes:
- `"duplicate"`: Replace computed hint channels with copies of `scaled_target`
- `"learned_conv"`: Replace computed hints with output of a learned `nn.Conv1d`

Add `hint_ablation_kernel_size` parameter (int, default=4).

For `"learned_conv"`:
- Create `nn.Conv1d(1, 1, kernel_size=K, stride=patch_size, padding=K-1, bias=False)` in `__init__`
- Initialize weights from Chebyshev d=4 coefficients (zero-pad if K > 4)
- In forward: apply conv to scaled_target, compute residual, use as hint channel
- Conv operates per-variate (reshape to treat batch*variate as batch dim)

### SLURM scripts

3 training scripts + 3 eval scripts (duplicate, learned_4tap, learned_16tap).

### Expected outcomes

- If `duplicate` matches hint d=4: benefit is from extra capacity, not filter content
- If `learned_conv` matches/beats hint d=4: Chebyshev structure isn't special
- If hint d=4 beats both: the specific fixed polynomial structure matters
