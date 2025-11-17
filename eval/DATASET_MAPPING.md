# Dataset Name Mapping

This document explains how dataset names in `forecast_datasets.xlsx` map to actual cached dataset paths.

## Mapping Logic

The `read_datasets_config.py` script maps display names to cached dataset paths using:

1. **Explicit mapping** for datasets with non-standard names (see table below)
2. **Auto-generation** for standard datasets (lowercase, underscores, no special chars)

## Explicit Mappings (Special Cases)

| Display Name (Excel) | Cached Dataset Path | Reason |
|---------------------|---------------------|---------|
| Aus. Elec. Demand | `australian_electricity_demand` | Abbreviation expansion |
| Australia Weather | `weather` | Short form |
| Bitcoin | `bitcoin_with_missing` | Has `_with_missing` suffix |
| Carparts | `car_parts_with_missing` | Space + suffix |
| CIF 2016 | `cif_2016_12` | 12-month version (vs 6-month) |
| KDD Cup 2018 | `kdd_cup_2018_with_missing` | Has suffix |
| M3 Monthly | `monash_m3_monthly` | Has `monash_` prefix |
| M3 Other | `monash_m3_other` | Has `monash_` prefix |
| NN5 Daily | `nn5_daily_with_missing` | Has suffix |
| NN5 Weekly | `nn5_weekly` | Standard |
| Rideshare | `rideshare_with_missing` | Has suffix |
| Saugeen River Flow | `saugeenday` | Different name |
| Sunspot | `sunspot_with_missing` | Has suffix |
| Temperature Rain | `temperature_rain_with_missing` | Has suffix |
| Vehicle Trips | `vehicle_trips_with_missing` | Has suffix |

## Auto-Generated Mappings

These datasets use standard naming (display name → lowercase with underscores):

- M1 Monthly → `m1_monthly`
- M4 Monthly → `m4_monthly`
- M4 Weekly → `m4_weekly`
- M4 Daily → `m4_daily`
- M4 Hourly → `m4_hourly`
- Tourism Quarterly → `tourism_quarterly`
- Tourism Monthly → `tourism_monthly`
- Pedestrian Counts → `pedestrian_counts`
- FRED-MD → `fred_md`
- Traffic Hourly → `traffic_hourly`
- Traffic Weekly → `traffic_weekly`
- Hospital → `hospital`
- COVID Deaths → `covid_deaths`
- US Births → `us_births`

## Validation

All 29 datasets in `forecast_datasets.xlsx` have been validated to exist in:
```
/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1/
```

Run validation test:
```bash
cd /scratch/gpfs/EHAZAN/jh1161/eval
python3 -c "
import subprocess, json
result = subprocess.run(['python3', 'read_datasets_config.py'], capture_output=True, text=True)
data = json.loads(result.stdout)
cache = set(subprocess.run(['ls', '/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1'],
            capture_output=True, text=True).stdout.strip().split('\n'))
missing = [d for d in data if d['dataset_name'] not in cache]
print(f'Total datasets: {len(data)}')
print(f'Valid: {len(data) - len(missing)}')
print(f'Missing: {len(missing)}')
"
```

Expected output: `Total datasets: 29, Valid: 29, Missing: 0`

## Dataset Path Format

Cached datasets follow the format:
```
/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1/<dataset_name>/
```

Each dataset directory contains:
- Metadata files
- Time series data in Arrow format
- Train/test splits (if applicable)
