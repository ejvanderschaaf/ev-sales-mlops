import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_ingest import ingest
from src.data_validation import validate

#  Minimal synthetic data

SAMPLE_DATA = {
    "country": ["China"] * 4 + ["USA"] * 4,
    "brand": ["BYD"] * 4 + ["Tesla"] * 4,
    "drivetrain_type": ["BEV+PHEV"] * 8,
    "year": [2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021],
    "month": [1, 2, 3, 4, 1, 2, 3, 4],
    "units_sold": [50000, 55000, 58000, 62000, 10000, 12000, 11000, 13000],
    "frequency": ["monthly_est"] * 8,
    "source": ["brand_reports"] * 8,
    "trend_byd": [10.0] * 8,
    "trend_ev_charging": [20.0] * 8,
    "trend_tesla": [60.0] * 8,
    "trend_electric_car": [40.0] * 8,
    "trend_electric_vehicle": [35.0] * 8,
    "slow_chargers_cumulative": [600000.0] * 8,
    "fast_chargers_cumulative": [100000.0] * 8,
    "total_chargers_cumulative": [700000.0] * 8,
    "gasoline_price_usd_per_liter": [0.80] * 8,
    "units_sold_lag1": [45000.0, 50000.0, 55000.0, 58000.0,
                        9000.0, 10000.0, 12000.0, 11000.0],
    "units_sold_lag3": [0.0] * 8,
    "units_sold_lag12": [0.0] * 8,
    "units_sold_rolling3": [50000.0, 52500.0, 54333.0, 58333.0,
                             10000.0, 11000.0, 11000.0, 12000.0],
    "units_sold_yoy_growth": [np.nan] * 8,
    "quarter": [1, 1, 1, 2, 1, 1, 1, 2],
    "date": ["2021-01-01"] * 8,
    "date_str": ["2021-01"] * 8,
}


@pytest.fixture
def tmp_raw(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    staged_dir = tmp_path / "data" / "staged"
    staged_dir.mkdir(parents=True)
    raw_path = raw_dir / "ev_demand_dataset.csv"
    pd.DataFrame(SAMPLE_DATA).to_csv(raw_path, index=False)
    return tmp_path


#  Ingest tests

def test_ingest_returns_dataframe(tmp_raw):
    df = ingest(
        raw_path=str(tmp_raw / "data/raw/ev_demand_dataset.csv"),
        staged_path=str(tmp_raw / "data/staged/data.csv"),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_ingest_drops_metadata_columns(tmp_raw):
    df = ingest(
        raw_path=str(tmp_raw / "data/raw/ev_demand_dataset.csv"),
        staged_path=str(tmp_raw / "data/staged/data.csv"),
    )
    for col in ["frequency", "source", "date_str", "units_sold_yoy_growth"]:
        assert col not in df.columns, f"Expected {col} to be dropped"


def test_ingest_no_nulls_in_critical_columns(tmp_raw):
    df = ingest(
        raw_path=str(tmp_raw / "data/raw/ev_demand_dataset.csv"),
        staged_path=str(tmp_raw / "data/staged/data.csv"),
    )
    for col in ["units_sold", "units_sold_lag1"]:
        assert df[col].isnull().sum() == 0, f"Unexpected nulls in {col}"


def test_ingest_units_sold_non_negative(tmp_raw):
    df = ingest(
        raw_path=str(tmp_raw / "data/raw/ev_demand_dataset.csv"),
        staged_path=str(tmp_raw / "data/staged/data.csv"),
    )
    assert (df["units_sold"] >= 0).all()


def test_ingest_saves_csv(tmp_raw):
    staged_path = str(tmp_raw / "data/staged/data.csv")
    ingest(
        raw_path=str(tmp_raw / "data/raw/ev_demand_dataset.csv"),
        staged_path=staged_path,
    )
    assert os.path.exists(staged_path)


#  Validation tests

def test_validation_passes_on_clean_data(tmp_raw):
    staged_path = str(tmp_raw / "data/staged/data.csv")
    ingest(
        raw_path=str(tmp_raw / "data/raw/ev_demand_dataset.csv"),
        staged_path=staged_path,
    )
    # Patch expected countries to match sample data
    import src.data_validation as dv
    original = dv.EXPECTED_COUNTRIES
    dv.EXPECTED_COUNTRIES = {"China", "USA"}
    try:
        report = validate(
            staged_path=staged_path,
            report_path=str(tmp_raw / "data/staged/validation_report.json"),
        )
        assert report["passed"] is True
    finally:
        dv.EXPECTED_COUNTRIES = original


def test_validation_fails_on_negative_units(tmp_raw):
    staged_path = str(tmp_raw / "data/staged/data.csv")
    df = ingest(
        raw_path=str(tmp_raw / "data/raw/ev_demand_dataset.csv"),
        staged_path=staged_path,
    )
    df.loc[0, "units_sold"] = -999
    df.to_csv(staged_path, index=False)

    import src.data_validation as dv
    original = dv.EXPECTED_COUNTRIES
    dv.EXPECTED_COUNTRIES = {"China", "USA"}
    try:
        with pytest.raises(ValueError):
            validate(
                staged_path=staged_path,
                report_path=str(tmp_raw / "data/staged/validation_report.json"),
            )
    finally:
        dv.EXPECTED_COUNTRIES = original
