#!/usr/bin/env python3
"""Check all Rideshare samples for NaN issues."""

import numpy as np
from uni2ts.eval_util.data import get_gluonts_test_dataset

# Load the dataset
print("Loading Rideshare dataset...")
test_data, metadata = get_gluonts_test_dataset(
    dataset_name='rideshare_with_missing',
    prediction_length=168,
    use_lotsa_cache=True
)

test_list = list(test_data)
print(f"Total samples: {len(test_list)}\n")

valid_samples = 0
total_nans_in_labels = 0
max_valid_label_len = 0

for i, item in enumerate(test_list):
    if isinstance(item, tuple):
        input_dict, label_dict = item
    else:
        input_dict = item.input
        label_dict = item.label

    label_nans = np.sum(np.isnan(label_dict['target']))
    total_nans_in_labels += label_nans

    valid_len = 168 - label_nans
    if valid_len > max_valid_label_len:
        max_valid_label_len = valid_len

    if label_nans == 0:
        valid_samples += 1

print(f"Samples with NO NaNs in label: {valid_samples}/{len(test_list)}")
print(f"Total NaNs in all labels: {total_nans_in_labels}")
print(f"Average NaNs per label: {total_nans_in_labels / len(test_list):.1f}")
print(f"Max valid (non-NaN) values in any label: {max_valid_label_len}")

# Check if the issue is prediction_length mismatch
print(f"\nPrediction length setting: {metadata.prediction_length}")
print(f"Label shape: {label_dict['target'].shape}")
