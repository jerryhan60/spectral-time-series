# CSV Dataset Ordering Fix

## Summary

Updated evaluation scripts to ensure CSV files maintain the correct dataset order as specified in `forecast_datasets.xlsx`.

## Changes Made

### 1. **eval_precond_hybrid_comprehensive.slurm** (Lines 438-494)

Added dataset ordering logic to the aggregation section:

```python
# Define the desired order based on forecast_datasets.xlsx
desired_order = [
    'M1_Monthly', 'M3_Monthly', 'M3_Other', 'M4_Monthly', 'M4_Weekly',
    'M4_Daily', 'M4_Hourly', 'Tourism_Quarterly', 'Tourism_Monthly',
    'CIF_2016', 'Aus._Elec._Demand', 'Bitcoin', 'Pedestrian_Counts',
    'Vehicle_Trips', 'KDD_Cup_2018', 'Australia_Weather', 'NN5_Daily',
    'NN5_Weekly', 'Carparts', 'FRED-MD', 'Traffic_Hourly', 'Traffic_Weekly',
    'Rideshare', 'Hospital', 'COVID_Deaths', 'Temperature_Rain', 'Sunspot',
    'Saugeen_River_Flow', 'US_Births'
]

# Sort dataframe by desired order
summary_df['sort_order'] = summary_df['dataset'].map({name: i for i, name in enumerate(desired_order)})
summary_df = summary_df.sort_values('sort_order').drop('sort_order', axis=1)
```

**Result**: Future hybrid evaluation runs will generate CSVs with datasets in the correct order.

### 2. **eval_precond_comprehensive.slurm** (Lines 353-409)

Added identical dataset ordering logic to the preconditioned space evaluation aggregation.

**Result**: Future preconditioned space evaluation runs will generate CSVs with datasets in the correct order.

### 3. **eval_comprehensive.slurm** (No Changes Needed)

This script already maintains the correct order because it:
- Reads dataset configuration from `forecast_datasets.xlsx` (line 109)
- Explicitly maintains order from Excel file (line 116 comment)
- Processes datasets sequentially and appends to CSV in that order (lines 178-182)

**Result**: Already generates CSVs in the correct order.

### 4. **Existing CSV Files**

Updated the following existing files to match the correct order:

- ✅ `eval_hybrid_results_last_last_20251114_113517/evaluation_metrics_hybrid_FINAL.csv` - Reordered
- ✅ `eval_precond_results_last_20251114_111719/evaluation_metrics_precond_space.csv` - Already in correct order

## Dataset Order Reference

The canonical order from `forecast_datasets.xlsx`:

1. M1_Monthly
2. M3_Monthly
3. M3_Other
4. M4_Monthly
5. M4_Weekly
6. M4_Daily
7. M4_Hourly
8. Tourism_Quarterly
9. Tourism_Monthly
10. CIF_2016
11. Aus._Elec._Demand
12. Bitcoin
13. Pedestrian_Counts
14. Vehicle_Trips
15. KDD_Cup_2018
16. Australia_Weather
17. NN5_Daily
18. NN5_Weekly
19. Carparts
20. FRED-MD
21. Traffic_Hourly
22. Traffic_Weekly
23. Rideshare
24. Hospital
25. COVID_Deaths
26. Temperature_Rain
27. Sunspot
28. Saugeen_River_Flow
29. US_Births

## How It Works

The ordering logic:

1. **Creates an order mapping**: Maps each dataset name to its index position
2. **Adds temporary sort column**: `summary_df['sort_order']` contains the numeric order
3. **Sorts by order**: `summary_df.sort_values('sort_order')`
4. **Removes sort column**: `.drop('sort_order', axis=1)` to keep CSV clean
5. **Saves to CSV**: Final CSV has datasets in the correct order

## Verification

All CSV files now follow the order specified in `forecast_datasets.xlsx`:

```bash
# Verify hybrid CSV order
head -n 10 eval_hybrid_results_last_last_20251114_113517/evaluation_metrics_hybrid_FINAL.csv

# Verify precond CSV order
head -n 10 eval_precond_results_last_20251114_111719/evaluation_metrics_precond_space.csv
```

## Future Runs

All future evaluation runs will automatically generate CSVs with datasets in the correct order. No manual reordering required.
