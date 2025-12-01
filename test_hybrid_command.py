#!/usr/bin/env python3
"""
Quick test to verify the hybrid evaluation command is constructed correctly.
"""

# Simulate the command construction
base_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/checkpoints/last.ckpt"
precond_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/precond_chebyshev_d5_20251117_204835/checkpoints/last.ckpt"

base_model_name = "pretrain_run_20251020_205126"
precond_model_name = "precond_chebyshev_d5_20251117_204835"
dataset_name = "monash_m3_monthly"
dataset_patch_size = 32
context_length = 1000
precond_type = 'chebyshev'
precond_degree = 5
batch_size = 32
pred_length = 18

# Build command (as done in comprehensive_evaluation.py)
cmd = [
    'python', '-m', 'cli.eval_precond_hybrid',
    f'run_name=eval_hybrid_{base_model_name}_{precond_model_name}_{dataset_name}',
    # Base model configuration (override checkpoint and params)
    f'base_model.checkpoint_path={base_model_path}',
    f'base_model.patch_size={dataset_patch_size}',
    f'base_model.context_length={context_length}',
    # Preconditioned model configuration (override checkpoint and params)
    f'precond_model.checkpoint_path={precond_model_path}',
    f'precond_model.patch_size={dataset_patch_size}',
    f'precond_model.context_length={context_length}',
    f'precond_model.precondition_type={precond_type}',
    f'precond_model.precondition_degree={precond_degree}',
    # Data and batch configuration
    f'batch_size={batch_size}',
    'data=monash_cached',
    f'data.dataset_name={dataset_name}',
    f'data.prediction_length={pred_length}'
]

print("Generated hybrid evaluation command:")
print(" ".join(cmd))
print("\n" + "="*80)
print("Key differences from previous version:")
print("="*80)
print("✓ Removed: 'base_model=moirai_lightning_ckpt' (already in defaults)")
print("✓ Removed: 'precond_model=moirai_precond_ckpt_no_reverse' (already in defaults)")
print("✓ Now only overriding specific fields within the configs")
print("✓ default_hybrid.yaml updated to use moirai_lightning_ckpt for base_model")
print("\nThis should resolve the Hydra struct mode error!")
