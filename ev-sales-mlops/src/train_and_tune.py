import argparse
import json
import logging
import os

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGED_PATH = "data/staged/data.csv"
MODEL_PATH = "models/model.pkl"
PARAMS_OUT_PATH = "models/best_params.json"

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

PARAM_GRID = {
    "regressor__n_estimators": [100, 200, 300, 500],
    "regressor__max_depth": [3, 4, 5, 6],
    "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
    "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


def load_and_split(staged_path: str):
    df = pd.read_csv(staged_path)
    df = df.sort_values(["country", "brand", "year", "month"]).reset_index(drop=True)
    train = df[df["year"] < 2023].copy()
    test = df[df["year"] >= 2023].copy()
    X_train = train[CAT_FEATURES + NUM_FEATURES]
    y_train = train["units_sold"]
    X_test = test[CAT_FEATURES + NUM_FEATURES]
    y_test = test["units_sold"]
    return X_train, y_train, X_test, y_test


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            random_state=42, n_jobs=-1, objective="reg:squarederror"
        )),
    ])


def train(
    staged_path: str = STAGED_PATH,
    model_path: str = MODEL_PATH,
    params_out: str = PARAMS_OUT_PATH,
    n_iter: int = 20,
    cv_splits: int = 5,
    sanity: bool = False,
):
    X_train, y_train, X_test, y_test = load_and_split(staged_path)

    if sanity:
        # Fast CI sanity check — one country, capped iterations
        mask = X_train["country"] == X_train["country"].iloc[0]
        X_train, y_train = X_train[mask], y_train[mask]
        n_iter, cv_splits = 3, 2
        logger.info("Sanity mode: fast CI check on single country")

    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    pipeline = build_pipeline()
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        pipeline,
        PARAM_GRID,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    mlflow.set_experiment("ev-sales-forecast")
    with mlflow.start_run():
        logger.info("Starting hyperparameter search...")
        search.fit(X_train, y_train)

        best_params = search.best_params_
        cv_rmse = -search.best_score_

        mlflow.log_params(
            {k.replace("regressor__", ""): v for k, v in best_params.items()}
        )
        mlflow.log_metric("cv_rmse", cv_rmse)

        y_pred = search.predict(X_test)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100)

        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mape", test_mape)

        logger.info(
            f"CV RMSE: {cv_rmse:.2f} | Test RMSE: {test_rmse:.2f} | "
            f"Test MAPE: {test_mape:.2f}%"
        )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(search.best_estimator_, model_path)
        mlflow.log_artifact(model_path)

        with open(params_out, "w") as f:
            json.dump(best_params, f, indent=2)

        logger.info(f"Model saved → {model_path}")

    return search.best_estimator_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity", action="store_true", help="Fast sanity-check mode for CI"
    )
    args = parser.parse_args()
    train(sanity=args.sanity)
