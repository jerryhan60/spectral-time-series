#!/usr/bin/env python3
"""Debug script to check Rideshare dataset and predictions."""

import numpy as np
from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.transform import PolynomialPrecondition

# Load the dataset
print("Loading Rideshare dataset...")
test_data, metadata = get_gluonts_test_dataset(
    dataset_name='rideshare_with_missing',
    prediction_length=168,
    use_lotsa_cache=True
)

print(f"Metadata:")
print(f"  Prediction length: {metadata.prediction_length}")
print(f"  Frequency: {metadata.freq}")
print(f"  Target dim: {metadata.target_dim}")

# Check a few samples
test_list = list(test_data)
print(f"\nNumber of test samples: {len(test_list)}")

# Initialize preconditioner
preconditioner = PolynomialPrecondition(
    polynomial_type="chebyshev",
    degree=5,
    target_field="target",
    enabled=True,
    store_original=False,
)

# Check first few samples
for i in range(min(3, len(test_list))):
    item = test_list[i]
    # Handle tuple format
    if isinstance(item, tuple):
        input_dict, label_dict = item
    else:
        input_dict = item.input
        label_dict = item.label

    print(f"\n--- Sample {i} ---")
    print(f"Input target shape: {input_dict['target'].shape}")
    print(f"Label target shape: {label_dict['target'].shape}")
    print(f"Input target stats: min={np.min(input_dict['target']):.3f}, max={np.max(input_dict['target']):.3f}, mean={np.mean(input_dict['target']):.3f}")
    print(f"Label target stats: min={np.min(label_dict['target']):.3f}, max={np.max(label_dict['target']):.3f}, mean={np.mean(label_dict['target']):.3f}")

    # Check for zeros or NaNs
    input_zeros = np.sum(input_dict['target'] == 0)
    label_zeros = np.sum(label_dict['target'] == 0)
    input_nans = np.sum(np.isnan(input_dict['target']))
    label_nans = np.sum(np.isnan(label_dict['target']))

    print(f"Input zeros: {input_zeros}, NaNs: {input_nans}")
    print(f"Label zeros: {label_zeros}, NaNs: {label_nans}")

    # Apply preconditioning
    full_target = np.concatenate([input_dict["target"], label_dict["target"]], axis=0)
    data_entry = {"target": full_target}
    preconditioned_entry = preconditioner(data_entry)
    preconditioned_full = preconditioned_entry["target"]

    # Split back
    input_len = len(input_dict["target"])
    preconditioned_label = preconditioned_full[input_len:]

    print(f"Preconditioned label stats: min={np.min(preconditioned_label):.3f}, max={np.max(preconditioned_label):.3f}, mean={np.mean(preconditioned_label):.3f}")
    precond_zeros = np.sum(preconditioned_label == 0)
    precond_nans = np.sum(np.isnan(preconditioned_label))
    print(f"Preconditioned label zeros: {precond_zeros}, NaNs: {precond_nans}")
