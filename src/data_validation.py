import pandas as pd
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGED_PATH = "data/staged/data.csv"
REPORT_PATH = "data/staged/validation_report.json"

EXPECTED_COLUMNS = [
    "country", "brand", "drivetrain_type", "year", "month", "units_sold",
    "quarter", "trend_byd", "trend_ev_charging", "trend_tesla",
    "trend_electric_car", "trend_electric_vehicle",
    "slow_chargers_cumulative", "fast_chargers_cumulative",
    "total_chargers_cumulative", "gasoline_price_usd_per_liter",
    "units_sold_lag1", "units_sold_lag3", "units_sold_lag12",
    "units_sold_rolling3",
]

EXPECTED_COUNTRIES = {
    "China", "USA", "Germany", "United Kingdom", "France", "Norway", "India", "Netherlands"
}


def validate(staged_path=STAGED_PATH, report_path=REPORT_PATH):
    df = pd.read_csv(staged_path)
    report = {"passed": True, "checks": {}}

    # 1. Schema
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    report["checks"]["schema"] = {
        "passed": len(missing_cols) == 0,
        "missing_columns": missing_cols,
    }

    # 2. No nulls in critical columns
    critical = ["units_sold", "units_sold_lag1", "country", "brand", "year", "month"]
    null_counts = df[critical].isnull().sum().to_dict()
    report["checks"]["no_nulls_critical"] = {
        "passed": all(v == 0 for v in null_counts.values()),
        "null_counts": null_counts,
    }

    # 3. Country coverage
    actual_countries = set(df["country"].unique())
    report["checks"]["country_coverage"] = {
        "passed": actual_countries == EXPECTED_COUNTRIES,
        "found": sorted(actual_countries),
        "expected": sorted(EXPECTED_COUNTRIES),
    }

    # 4. Non-negative target
    neg_count = int((df["units_sold"] < 0).sum())
    report["checks"]["units_sold_non_negative"] = {
        "passed": neg_count == 0,
        "negative_count": neg_count,
    }

    # 5. Date range
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    report["checks"]["date_range"] = {
        "passed": min_year >= 2019 and max_year <= 2023,
        "min_year": min_year,
        "max_year": max_year,
    }

    # 6. Distribution summary
    report["checks"]["units_sold_stats"] = df["units_sold"].describe().to_dict()

    # Overall result
    report["passed"] = all(
        v["passed"]
        for v in report["checks"].values()
        if isinstance(v, dict) and "passed" in v
    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    status = "PASSED" if report["passed"] else "FAILED"
    logger.info(f"Validation {status} — see {report_path}")

    if not report["passed"]:
        raise ValueError(f"Data validation failed. Check {report_path}")

    return report


if __name__ == "__main__":
    validate()
