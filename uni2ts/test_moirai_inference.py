#!/usr/bin/env python3
"""Quick inference test for Moirai small model"""

import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

print("=" * 60)
print("Moirai Small Model Inference Test")
print("=" * 60)

# Check GPU availability
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ No GPU available, using CPU")

print("\nLoading sample data...")
# Create simple synthetic time series data
dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
df = pd.DataFrame({
    'series_1': range(500),
    'series_2': [x * 2 for x in range(500)]
}, index=dates)

# Convert to GluonTS dataset
ds = PandasDataset(dict(df))

# Split into train/test
TEST = 50
train, test_template = split(ds, offset=-TEST)

print(f"✓ Data loaded: {len(df)} time steps, {len(df.columns)} series")

print("\nLoading Moirai-1.1-R-small model from HuggingFace...")
# Model parameters
PDT = 20  # prediction length
CTX = 100  # context length
PSZ = "auto"  # patch size
BSZ = 4  # batch size

model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

print("✓ Model loaded successfully")

# Create predictor
print("\nCreating predictor and generating forecasts...")
predictor = model.create_predictor(batch_size=BSZ)

# Generate test data
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=2,  # Just 2 windows for quick test
    distance=PDT,
)

# Run inference
forecasts = list(predictor.predict(test_data.input))

print(f"✓ Generated {len(forecasts)} forecasts")
print(f"  Forecast shape: {forecasts[0].mean.shape}")
print(f"  Prediction length: {PDT}")

print("\n" + "=" * 60)
print("✓ Inference test completed successfully!")
print("=" * 60)
