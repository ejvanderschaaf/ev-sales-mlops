"""
scripts/deploy_sagemaker.py

Creates or updates a SageMaker endpoint with the latest ECR image.
Called by the CD workflow after Docker push.
"""
import logging
import os
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGION = "us-east-2"
ACCOUNT_ID = os.environ["AWS_ACCOUNT_ID"]
ECR_IMAGE = (
    f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/ev-sales-mlops:latest"
)
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole"
MODEL_NAME = "ev-sales-xgboost"
ENDPOINT_NAME = "ev-sales-endpoint"


def deploy():
    sm = boto3.client("sagemaker", region_name=REGION)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    config_name = f"ev-sales-config-{timestamp}"

    #  Create model
    MODEL_DATA_URL = "s3://ev-sales-mlops/model/model.tar.gz"

    try:
        sm.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={
                "Image": ECR_IMAGE,
                "ModelDataUrl": MODEL_DATA_URL,
                "Environment": {"MODEL_PATH": "/opt/ml/model/model.pkl"},
            },
            ExecutionRoleArn=ROLE_ARN,
        )
        logger.info(f"Created SageMaker model: {MODEL_NAME}")

    #  Create endpoint config
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": MODEL_NAME,
            "InstanceType": "ml.t2.medium",
            "InitialInstanceCount": 1,
        }],
    )
    logger.info(f"Created endpoint config: {config_name}")

    #  Create or update endpoint
    try:
        sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
        logger.info(f"Creating new endpoint: {ENDPOINT_NAME}")
    except ClientError as e:
        if "already exists" in str(e) or "already existing" in str(e):
            sm.update_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=config_name,
            )
            logger.info(f"Updating existing endpoint: {ENDPOINT_NAME}")
        else:
            raise

    logger.info("Deployment complete.")


if __name__ == "__main__":
    deploy()
