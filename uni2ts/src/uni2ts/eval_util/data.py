from functools import partial
from typing import NamedTuple

import gluonts
import numpy as np
from gluonts.dataset.common import _FileDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import TestData, split

from uni2ts.data.builder.lotsa_v1.gluonts import get_dataset

from ._hf_dataset import HFDataset
from ._lsf_dataset import LSFDataset
from ._pf_dataset import generate_pf_dataset, pf_load_func_map

gluonts.dataset.repository.dataset_recipes |= {
    k: partial(generate_pf_dataset, dataset_name=k) for k in pf_load_func_map.keys()
}


class MetaData(NamedTuple):
    freq: str
    target_dim: int
    prediction_length: int
    feat_dynamic_real_dim: int = 0
    past_feat_dynamic_real_dim: int = 0
    split: str = "test"


def get_gluonts_val_dataset(
    dataset_name: str,
    prediction_length: int = None,
    mode: str = None,
    regenerate: bool = False,
) -> tuple[TestData, MetaData]:
    # Comprehensive mapping of dataset names to prediction lengths
    # Based on eval_confs/forecast_datasets.xlsx
    default_prediction_lengths = {
        "m1_monthly": 18,
        "m3_monthly": 18,
        "monash_m3_monthly": 18,
        "m3_other": 8,
        "monash_m3_other": 8,
        "m4_monthly": 18,
        "m4_weekly": 13,
        "m4_daily": 14,
        "m4_hourly": 48,
        "tourism_monthly": 24,
        "tourism_quarterly": 8,
        "cif_2016": 12,
        "cif_2016_12": 12,
        "nn5_daily_with_missing": 56,
        "nn5_weekly": 8,
        "traffic_weekly": 8,
        "traffic_hourly": 168,
        "us_births": 30,
        "saugeenday": 30,
        "saugeen_river_flow": 30,
        "solar_weekly": 5,
        "car_parts_with_missing": 12,
        "fred_md": 12,
        "covid_deaths": 30,
        "hospital": 12,
        "kdd_cup_2018_with_missing": 48,
        "weather": 30,
        "australia_weather": 30,
        "rideshare_with_missing": 168,
        "bitcoin_with_missing": 30,
        "pedestrian_counts": 24,
        "vehicle_trips_with_missing": 30,
        "australian_electricity_demand": 336,
        "temperature_rain": 30,
        "temperature_rain_with_missing": 30,
        "sunspot_with_missing": 30,
    }
    if prediction_length is None and dataset_name in default_prediction_lengths:
        prediction_length = default_prediction_lengths[dataset_name]

    dataset = get_dataset(
        dataset_name, prediction_length=prediction_length, regenerate=regenerate
    )

    prediction_length = prediction_length or dataset.metadata.prediction_length
    _, test_template = split(dataset.train, offset=-prediction_length)
    test_data = test_template.generate_instances(prediction_length)
    metadata = MetaData(
        freq=dataset.metadata.freq,
        target_dim=1,
        prediction_length=prediction_length,
        split="val",
    )
    return test_data, metadata


def get_gluonts_train_dataset(
    dataset_name: str,
    prediction_length: int = None,
    mode: str = None,
    regenerate: bool = False,
    use_lotsa_cache: bool = False,
) -> tuple[TestData, MetaData]:
    """
    Get train dataset from GluonTS repository for evaluation.
    This is used to evaluate on the training split as a sanity check.

    Args:
        dataset_name: Name of the dataset
        prediction_length: Prediction length (optional)
        mode: Mode (optional)
        regenerate: Whether to regenerate the dataset
        use_lotsa_cache: If True, use cached datasets from lotsa_v1 folder
                        instead of downloading from GluonTS repository.
                        Set to True when running on compute nodes without internet.
    """
    # Comprehensive mapping of dataset names to prediction lengths
    # Based on eval_confs/forecast_datasets.xlsx
    default_prediction_lengths = {
        "m1_monthly": 18,
        "m3_monthly": 18,
        "monash_m3_monthly": 18,
        "m3_other": 8,
        "monash_m3_other": 8,
        "m4_monthly": 18,
        "m4_weekly": 13,
        "m4_daily": 14,
        "m4_hourly": 48,
        "tourism_monthly": 24,
        "tourism_quarterly": 8,
        "cif_2016": 12,
        "cif_2016_12": 12,
        "nn5_daily_with_missing": 56,
        "nn5_weekly": 8,
        "traffic_weekly": 8,
        "traffic_hourly": 168,
        "us_births": 30,
        "saugeenday": 30,
        "saugeen_river_flow": 30,
        "solar_weekly": 5,
        "car_parts_with_missing": 12,
        "fred_md": 12,
        "covid_deaths": 30,
        "hospital": 12,
        "kdd_cup_2018_with_missing": 48,
        "weather": 30,
        "australia_weather": 30,
        "rideshare_with_missing": 168,
        "bitcoin_with_missing": 30,
        "pedestrian_counts": 24,
        "vehicle_trips_with_missing": 30,
        "australian_electricity_demand": 336,
        "temperature_rain": 30,
        "temperature_rain_with_missing": 30,
        "sunspot_with_missing": 30,
    }
    if prediction_length is None and dataset_name in default_prediction_lengths:
        prediction_length = default_prediction_lengths[dataset_name]

    if use_lotsa_cache:
        # Use cached dataset from lotsa_v1 folder (no internet required)
        from pathlib import Path
        import datasets as hf_datasets
        import pandas as pd

        lotsa_path = Path(__file__).parent.parent.parent / "data" / "lotsa_v1" / dataset_name
        if not lotsa_path.exists():
            # Fallback to absolute path
            lotsa_path = Path("/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1") / dataset_name

        if not lotsa_path.exists():
            raise FileNotFoundError(
                f"Cached dataset not found at {lotsa_path}. "
                f"Available datasets: run 'ls {lotsa_path.parent}'"
            )

        # Load HuggingFace dataset from disk
        hf_dataset = hf_datasets.load_from_disk(str(lotsa_path))

        # Get a sample item to extract metadata
        sample = hf_dataset[0]
        freq = sample['freq']

        # Default prediction lengths by dataset name/frequency
        if prediction_length is None:
            if 'yearly' in dataset_name:
                prediction_length = 4
            elif 'quarterly' in dataset_name:
                prediction_length = 8
            elif 'monthly' in dataset_name:
                prediction_length = 18
            else:
                prediction_length = 24

        # Convert HF dataset to GluonTS-compatible format
        def hf_to_gluonts_generator():
            for item in hf_dataset:
                # Create timestamp with frequency info
                ts = pd.Timestamp(item['start'])
                start = pd.Period(ts, freq=item['freq'])
                yield {
                    FieldName.ITEM_ID: item['item_id'],
                    FieldName.START: start,
                    FieldName.TARGET: np.array(item['target'], dtype=np.float32),
                }

        # Create iterable dataset
        gluonts_dataset = list(hf_to_gluonts_generator())

        # Create train split evaluation
        # We want to evaluate on a portion of the train data that the model saw during training
        # The model was trained on the TRAIN split (everything except last prediction_length)
        # So we evaluate on the last prediction_length of the TRAIN split
        # This is: data[-(2*pred_len) : -pred_len]

        # First, filter out series that are too short
        min_length = 3 * prediction_length  # Need enough data for meaningful train/test split
        filtered_dataset = [
            item for item in gluonts_dataset
            if len(item[FieldName.TARGET]) >= min_length
        ]

        if not filtered_dataset:
            raise ValueError(
                f"No time series in {dataset_name} are long enough "
                f"(min_length={min_length}) for train split evaluation"
            )

        # Split: everything up to -prediction_length is "training data"
        # The last prediction_length is the "test set" (held out)
        train_split, _ = split(filtered_dataset, offset=-prediction_length)

        # Now from the train split, take the last prediction_length for evaluation
        # This ensures we're evaluating on data the model saw during training
        _, train_eval_template = split(train_split, offset=-prediction_length)
        train_data = train_eval_template.generate_instances(prediction_length)

        metadata = MetaData(
            freq=freq,
            target_dim=1,
            prediction_length=prediction_length,
            split="train",
        )
        return train_data, metadata

    else:
        # Original behavior: download from GluonTS repository
        dataset = get_dataset(
            dataset_name, prediction_length=prediction_length, regenerate=regenerate
        )

        prediction_length = prediction_length or dataset.metadata.prediction_length
        # Use train split but take last prediction_length as validation
        _, train_template = split(dataset.train, offset=-prediction_length)
        train_data = train_template.generate_instances(prediction_length)
        metadata = MetaData(
            freq=dataset.metadata.freq,
            target_dim=1,
            prediction_length=prediction_length,
            split="train",
        )
        return train_data, metadata


def get_gluonts_test_dataset(
    dataset_name: str,
    prediction_length: int = None,
    mode: str = None,
    regenerate: bool = False,
    use_lotsa_cache: bool = False,
) -> tuple[TestData, MetaData]:
    """
    Get test dataset from GluonTS repository.

    Args:
        dataset_name: Name of the dataset
        prediction_length: Prediction length (optional)
        mode: Mode (optional)
        regenerate: Whether to regenerate the dataset
        use_lotsa_cache: If True, use cached datasets from lotsa_v1 folder
                        instead of downloading from GluonTS repository.
                        Set to True when running on compute nodes without internet.
    """
    # Comprehensive mapping of dataset names to prediction lengths
    # Based on eval_confs/forecast_datasets.xlsx
    default_prediction_lengths = {
        "m1_monthly": 18,
        "m3_monthly": 18,
        "monash_m3_monthly": 18,
        "m3_other": 8,
        "monash_m3_other": 8,
        "m4_monthly": 18,
        "m4_weekly": 13,
        "m4_daily": 14,
        "m4_hourly": 48,
        "tourism_monthly": 24,
        "tourism_quarterly": 8,
        "cif_2016": 12,
        "cif_2016_12": 12,
        "nn5_daily_with_missing": 56,
        "nn5_weekly": 8,
        "traffic_weekly": 8,
        "traffic_hourly": 168,
        "us_births": 30,
        "saugeenday": 30,
        "saugeen_river_flow": 30,
        "solar_weekly": 5,
        "car_parts_with_missing": 12,
        "fred_md": 12,
        "covid_deaths": 30,
        "hospital": 12,
        "kdd_cup_2018_with_missing": 48,
        "weather": 30,
        "australia_weather": 30,
        "rideshare_with_missing": 168,
        "bitcoin_with_missing": 30,
        "pedestrian_counts": 24,
        "vehicle_trips_with_missing": 30,
        "australian_electricity_demand": 336,
        "temperature_rain": 30,
        "temperature_rain_with_missing": 30,
        "sunspot_with_missing": 30,
    }
    if prediction_length is None and dataset_name in default_prediction_lengths:
        prediction_length = default_prediction_lengths[dataset_name]

    if use_lotsa_cache:
        # Use cached dataset from lotsa_v1 folder (no internet required)
        from pathlib import Path
        import datasets as hf_datasets
        import pandas as pd

        lotsa_path = Path(__file__).parent.parent.parent / "data" / "lotsa_v1" / dataset_name
        if not lotsa_path.exists():
            # Fallback to absolute path
            lotsa_path = Path("/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1") / dataset_name

        if not lotsa_path.exists():
            raise FileNotFoundError(
                f"Cached dataset not found at {lotsa_path}. "
                f"Available datasets: run 'ls {lotsa_path.parent}'"
            )

        # Load HuggingFace dataset from disk
        hf_dataset = hf_datasets.load_from_disk(str(lotsa_path))

        # Get a sample item to extract metadata
        sample = hf_dataset[0]
        freq = sample['freq']

        # Default prediction lengths by dataset name/frequency
        if prediction_length is None:
            if 'yearly' in dataset_name:
                prediction_length = 4
            elif 'quarterly' in dataset_name:
                prediction_length = 8
            elif 'monthly' in dataset_name:
                prediction_length = 18
            else:
                prediction_length = 24

        # Convert HF dataset to GluonTS-compatible format
        def hf_to_gluonts_generator():
            for item in hf_dataset:
                # Create timestamp with frequency info
                ts = pd.Timestamp(item['start'])
                start = pd.Period(ts, freq=item['freq'])
                yield {
                    FieldName.ITEM_ID: item['item_id'],
                    FieldName.START: start,
                    FieldName.TARGET: np.array(item['target'], dtype=np.float32),
                }

        # Create iterable dataset
        gluonts_dataset = list(hf_to_gluonts_generator())

        # Create test split
        _, test_template = split(gluonts_dataset, offset=-prediction_length)
        test_data = test_template.generate_instances(prediction_length)

        metadata = MetaData(
            freq=freq,
            target_dim=1,
            prediction_length=prediction_length,
            split="test",
        )
        return test_data, metadata

    else:
        # Original behavior: download from GluonTS repository
        dataset = get_dataset(
            dataset_name, prediction_length=prediction_length, regenerate=regenerate
        )

        prediction_length = prediction_length or dataset.metadata.prediction_length
        _, test_template = split(dataset.test, offset=-prediction_length)
        test_data = test_template.generate_instances(prediction_length)
        metadata = MetaData(
            freq=dataset.metadata.freq,
            target_dim=1,
            prediction_length=prediction_length,
            split="test",
        )
        return test_data, metadata


def get_lsf_val_dataset(
    dataset_name: str,
    prediction_length: int = 96,
    mode: str = "S",
) -> tuple[TestData, MetaData]:
    lsf_dataset = LSFDataset(dataset_name, mode=mode, split="val")
    dataset = _FileDataset(
        lsf_dataset, freq=lsf_dataset.freq, one_dim_target=lsf_dataset.target_dim == 1
    )
    _, test_template = split(dataset, offset=-lsf_dataset.length)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=lsf_dataset.length - prediction_length + 1,
        distance=1,
    )
    metadata = MetaData(
        freq=lsf_dataset.freq,
        target_dim=lsf_dataset.target_dim,
        prediction_length=prediction_length,
        past_feat_dynamic_real_dim=lsf_dataset.past_feat_dynamic_real_dim,
        split="val",
    )
    return test_data, metadata


def get_lsf_test_dataset(
    dataset_name: str,
    prediction_length: int = 96,
    mode: str = "S",
) -> tuple[TestData, MetaData]:
    lsf_dataset = LSFDataset(dataset_name, mode=mode, split="test")
    dataset = _FileDataset(
        lsf_dataset, freq=lsf_dataset.freq, one_dim_target=lsf_dataset.target_dim == 1
    )
    _, test_template = split(dataset, offset=-lsf_dataset.length)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=lsf_dataset.length - prediction_length + 1,
        distance=1,
    )
    metadata = MetaData(
        freq=lsf_dataset.freq,
        target_dim=lsf_dataset.target_dim,
        prediction_length=prediction_length,
        past_feat_dynamic_real_dim=lsf_dataset.past_feat_dynamic_real_dim,
        split="test",
    )
    return test_data, metadata


def get_custom_eval_dataset(
    dataset_name: str,
    offset: int,
    windows: int,
    distance: int,
    prediction_length: int,
    mode: None = None,
) -> tuple[TestData, MetaData]:
    hf_dataset = HFDataset(dataset_name)
    dataset = _FileDataset(
        hf_dataset, freq=hf_dataset.freq, one_dim_target=hf_dataset.target_dim == 1
    )
    _, test_template = split(dataset, offset=offset)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=windows,
        distance=distance,
    )
    metadata = MetaData(
        freq=hf_dataset.freq,
        target_dim=hf_dataset.target_dim,
        prediction_length=prediction_length,
        split="test",
    )
    return test_data, metadata
