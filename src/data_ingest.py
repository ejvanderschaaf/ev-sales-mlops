import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/ev_demand_dataset.csv"
STAGED_PATH = "data/staged/data.csv"


def ingest(raw_path=RAW_PATH, staged_path=STAGED_PATH):
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape}")

    # Drop metadata and redundant columns
    drop_cols = ["frequency", "source", "date_str", "date", "units_sold_yoy_growth"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Sort by series for proper time ordering
    df = df.sort_values(["country", "brand", "year", "month"]).reset_index(drop=True)

    # Forward-fill missing charging data (India 2019-2020 IEA reporting gap)
    charger_cols = [
        "slow_chargers_cumulative",
        "fast_chargers_cumulative",
        "total_chargers_cumulative",
    ]
    df[charger_cols] = (
        df.groupby(["country", "brand"])[charger_cols].ffill().bfill()
    )

    # Impute lag3/lag12/rolling3 NaNs with 0 (series start)
    impute_cols = ["units_sold_lag3", "units_sold_lag12", "units_sold_rolling3"]
    df[impute_cols] = df[impute_cols].fillna(0)

    # Drop rows missing the target or lag1 (first row of each series)
    df = df.dropna(subset=["units_sold", "units_sold_lag1"])

    os.makedirs(os.path.dirname(staged_path), exist_ok=True)
    df.to_csv(staged_path, index=False)
    logger.info(f"Staged data saved → {staged_path}, shape: {df.shape}")
    return df


if __name__ == "__main__":
    ingest()
