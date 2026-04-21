# EV Sales Forecasting — MLOps Pipeline

XGBoost regression model predicting monthly EV unit sales across 8 countries (2019–2023), deployed on AWS SageMaker with full CI/CD automation.

---

## Project Structure

```
ev-sales-mlops/
├─ data/
│  ├─ raw/                        # Raw CSV (DVC-tracked, not in Git)
│  └─ staged/                     # Cleaned data + validation report
├─ src/
│  ├─ data_ingest.py              # Load, clean, save staged data
│  ├─ data_validation.py          # Schema, nulls, distribution checks
│  ├─ train_and_tune.py           # XGBoost + RandomizedSearchCV + MLflow
│  └─ evaluate.py                 # RMSE, MAPE, R², per-country breakdown
├─ inference/
│  └─ predict.py                  # Flask server (/ping, /invocations)
├─ scripts/
│  └─ deploy_sagemaker.py         # Create/update SageMaker endpoint
├─ tests/
│  └─ test_pipeline.py            # pytest unit tests
├─ .github/workflows/
│  ├─ ci.yml                      # Lint + test + sanity train (non-main push)
│  └─ cd.yml                      # Full pipeline + Docker + deploy (main merge)
├─ Dockerfile
├─ dvc.yaml
├─ requirements.txt
└─ README.md
```

---

## Dataset

**EV Demand Dataset** — Kaggle ([link](https://www.kaggle.com/datasets/rrokon/ev-demand-dataset-sales-trends-and-infrastructure))

Monthly EV market data across China, USA, Germany, UK, France, Norway, India, Netherlands (2019–2023). Combines brand-level sales (BYD, Tesla, VW), Google Trends search interest, charging infrastructure counts, and gasoline prices.

Place the downloaded CSV at:
```
data/raw/ev_demand_dataset.csv
```

---

## Local Setup

```bash
# 1. Clone and install
git clone <your-repo-url>
cd ev-sales-mlops
pip install -r requirements.txt

# 2. Initialize DVC and configure S3 remote
git init
dvc init
dvc remote add -d myremote s3://ev-sales-mlops/dvc-store

# 3. Add raw data to DVC tracking
dvc add data/raw/ev_demand_dataset.csv
git add data/raw/ev_demand_dataset.csv.dvc .gitignore
git commit -m "Track raw data with DVC"
dvc push

# 4. Run the full pipeline
dvc repro

# 5. View MLflow experiment results
mlflow ui
# Open http://localhost:5000
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
# Build
docker build -t ev-sales-mlops .

# Run inference server locally
docker run -p 8080:8080 \
  -v $(pwd)/models:/opt/ml/model \
  ev-sales-mlops

# Test the endpoints
curl http://localhost:8080/ping

curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "country": "China", "brand": "BYD", "drivetrain_type": "BEV+PHEV",
    "year": 2023, "month": 6, "quarter": 2,
    "trend_byd": 15.0, "trend_ev_charging": 25.0, "trend_tesla": 60.0,
    "trend_electric_car": 40.0, "trend_electric_vehicle": 35.0,
    "slow_chargers_cumulative": 800000, "fast_chargers_cumulative": 150000,
    "total_chargers_cumulative": 950000, "gasoline_price_usd_per_liter": 0.85,
    "units_sold_lag1": 80000, "units_sold_lag3": 75000,
    "units_sold_lag12": 60000, "units_sold_rolling3": 78000
  }'
```

---

## AWS Deployment

### Prerequisites
- AWS CLI configured (`aws configure`)
- ECR repository created: `ev-sales-mlops`
- IAM role `SageMakerExecutionRole` with S3 and ECR access

### Push to ECR and Deploy

```bash
# Authenticate Docker with ECR
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin \
  <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com

# Build, tag, push
docker build -t ev-sales-mlops .
docker tag ev-sales-mlops:latest \
  <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/ev-sales-mlops:latest
docker push \
  <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/ev-sales-mlops:latest

# Deploy SageMaker endpoint
AWS_ACCOUNT_ID=<ACCOUNT_ID> python scripts/deploy_sagemaker.py
```

### Test the SageMaker Endpoint

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")
payload = {
    "country": "USA", "brand": "Tesla", "drivetrain_type": "BEV",
    "year": 2023, "month": 3, "quarter": 1,
    "trend_byd": 5.0, "trend_ev_charging": 30.0, "trend_tesla": 85.0,
    "trend_electric_car": 50.0, "trend_electric_vehicle": 45.0,
    "slow_chargers_cumulative": 90000, "fast_chargers_cumulative": 25000,
    "total_chargers_cumulative": 115000, "gasoline_price_usd_per_liter": 1.10,
    "units_sold_lag1": 15000, "units_sold_lag3": 14000,
    "units_sold_lag12": 12000, "units_sold_rolling3": 14500
}
response = runtime.invoke_endpoint(
    EndpointName="ev-sales-endpoint",
    ContentType="application/json",
    Body=json.dumps(payload),
)
print(json.loads(response["Body"].read()))
```

---

## CI/CD

| Trigger | Workflow | Steps |
|---|---|---|
| Push to any branch (not main) | `ci.yml` | Lint → unit tests → sanity train (1 country, 3 iterations) |
| Merge to `main` | `cd.yml` | Full `dvc repro` → Docker build + ECR push → SageMaker endpoint update |

### Required GitHub Secrets

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_ACCOUNT_ID` | 12-digit AWS account ID |

---

## Monitoring

Metrics tracked via MLflow (experiment: `ev-sales-forecast`):
- `cv_rmse` — cross-validation RMSE during HPO
- `test_rmse` — holdout RMSE on 2023 data
- `test_mape` — mean absolute percentage error
- Per-country breakdown in `models/metrics.json`

Run `mlflow ui` locally to compare experiments across runs.
