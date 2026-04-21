import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGED_PATH = "data/staged/data.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"

CAT_FEATURES = ["country", "brand", "drivetrain_type"]
NUM_FEATURES = [
    "year", "month", "quarter",
    "trend_byd", "trend_ev_charging", "trend_tesla",
    "trend_electric_car", "trend_electric_vehicle",
    "slow_chargers_cumulative", "fast_chargers_cumulative",
    "total_chargers_cumulative", "gasoline_price_usd_per_liter",
    "units_sold_lag1", "units_sold_lag3", "units_sold_lag12",
    "units_sold_rolling3",
]


def evaluate(
    staged_path: str = STAGED_PATH,
    model_path: str = MODEL_PATH,
    metrics_path: str = METRICS_PATH,
):
    df = pd.read_csv(staged_path)
    test = df[df["year"] >= 2023].copy()

    X_test = test[CAT_FEATURES + NUM_FEATURES]
    y_test = test["units_sold"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100)
    r2 = float(r2_score(y_test, y_pred))

    # Per-country breakdown
    test = test.copy()
    test["predicted"] = y_pred
    country_metrics = {}
    for country, grp in test.groupby("country"):
        country_metrics[country] = {
            "rmse": float(np.sqrt(mean_squared_error(grp["units_sold"], grp["predicted"]))),
            "mape": float(
                np.mean(np.abs((grp["units_sold"] - grp["predicted"]) / (grp["units_sold"] + 1)))
                * 100
            ),
        }

    metrics = {
        "test_rmse": rmse,
        "test_mape": mape,
        "test_r2": r2,
        "per_country": country_metrics,
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate()
