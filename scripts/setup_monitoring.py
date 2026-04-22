"""
scripts/setup_monitoring.py

Creates a CloudWatch dashboard and alert alarms for the SageMaker endpoint.
Run once after the endpoint is InService:
    AWS_ACCOUNT_ID=<id> python scripts/setup_monitoring.py
"""
import json
import logging
import os

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGION = "us-east-2"
ACCOUNT_ID = os.environ["AWS_ACCOUNT_ID"]
ENDPOINT_NAME = "ev-sales-endpoint"
VARIANT_NAME = "AllTraffic"
ALARM_SNS_ARN = os.environ.get("ALARM_SNS_ARN", "")  # optional email alerts


def create_dashboard(cw):
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "title": "Invocations per Minute",
                    "metrics": [[
                        "AWS/SageMaker",
                        "Invocations",
                        "EndpointName", ENDPOINT_NAME,
                        "VariantName", VARIANT_NAME,
                    ]],
                    "period": 60,
                    "stat": "Sum",
                    "region": REGION,
                    "view": "timeSeries",
                },
            },
            {
                "type": "metric",
                "x": 12, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "title": "Model Latency (ms)",
                    "metrics": [[
                        "AWS/SageMaker",
                        "ModelLatency",
                        "EndpointName", ENDPOINT_NAME,
                        "VariantName", VARIANT_NAME,
                    ]],
                    "period": 60,
                    "stat": "Average",
                    "region": REGION,
                    "view": "timeSeries",
                },
            },
            {
                "type": "metric",
                "x": 0, "y": 6, "width": 12, "height": 6,
                "properties": {
                    "title": "4XX Errors",
                    "metrics": [[
                        "AWS/SageMaker",
                        "Invocation4XXErrors",
                        "EndpointName", ENDPOINT_NAME,
                        "VariantName", VARIANT_NAME,
                    ]],
                    "period": 60,
                    "stat": "Sum",
                    "region": REGION,
                    "view": "timeSeries",
                },
            },
            {
                "type": "metric",
                "x": 12, "y": 6, "width": 12, "height": 6,
                "properties": {
                    "title": "5XX Errors",
                    "metrics": [[
                        "AWS/SageMaker",
                        "Invocation5XXErrors",
                        "EndpointName", ENDPOINT_NAME,
                        "VariantName", VARIANT_NAME,
                    ]],
                    "period": 60,
                    "stat": "Sum",
                    "region": REGION,
                    "view": "timeSeries",
                },
            },
            {
                "type": "metric",
                "x": 0, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "title": "CPU Utilization (%)",
                    "metrics": [[
                        "/aws/sagemaker/Endpoints",
                        "CPUUtilization",
                        "EndpointName", ENDPOINT_NAME,
                        "VariantName", VARIANT_NAME,
                    ]],
                    "period": 60,
                    "stat": "Average",
                    "region": REGION,
                    "view": "timeSeries",
                },
            },
            {
                "type": "metric",
                "x": 12, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "title": "Memory Utilization (%)",
                    "metrics": [[
                        "/aws/sagemaker/Endpoints",
                        "MemoryUtilization",
                        "EndpointName", ENDPOINT_NAME,
                        "VariantName", VARIANT_NAME,
                    ]],
                    "period": 60,
                    "stat": "Average",
                    "region": REGION,
                    "view": "timeSeries",
                },
            },
        ]
    }

    cw.put_dashboard(
        DashboardName="ev-sales-mlops",
        DashboardBody=json.dumps(dashboard_body),
    )
    logger.info("Dashboard created: ev-sales-mlops")


def create_alarms(cw):
    alarm_defaults = dict(
        Namespace="AWS/SageMaker",
        Dimensions=[
            {"Name": "EndpointName", "Value": ENDPOINT_NAME},
            {"Name": "VariantName", "Value": VARIANT_NAME},
        ],
        EvaluationPeriods=2,
        Period=60,
        TreatMissingData="notBreaching",
        AlarmActions=[ALARM_SNS_ARN] if ALARM_SNS_ARN else [],
    )

    # Alarm 1: High latency (>5 seconds average)
    cw.put_metric_alarm(
        **alarm_defaults,
        AlarmName="ev-sales-high-latency",
        AlarmDescription="Model latency exceeded 5000ms average over 2 minutes",
        MetricName="ModelLatency",
        Statistic="Average",
        Threshold=5000000,  # microseconds
        ComparisonOperator="GreaterThanThreshold",
    )
    logger.info("Alarm created: ev-sales-high-latency")

    # Alarm 2: 5XX errors
    cw.put_metric_alarm(
        **alarm_defaults,
        AlarmName="ev-sales-5xx-errors",
        AlarmDescription="5XX errors detected on inference endpoint",
        MetricName="Invocation5XXErrors",
        Statistic="Sum",
        Threshold=1,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
    )
    logger.info("Alarm created: ev-sales-5xx-errors")

    # Alarm 3: 4XX errors (bad requests / potential drift in input schema)
    cw.put_metric_alarm(
        **alarm_defaults,
        AlarmName="ev-sales-4xx-errors",
        AlarmDescription="4XX errors — possible input schema drift or bad requests",
        MetricName="Invocation4XXErrors",
        Statistic="Sum",
        Threshold=5,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
    )
    logger.info("Alarm created: ev-sales-4xx-errors")


def main():
    cw = boto3.client("cloudwatch", region_name=REGION)
    create_dashboard(cw)
    create_alarms(cw)
    dashboard_url = (
        f"https://{REGION}.console.aws.amazon.com/cloudwatch/home"
        f"?region={REGION}#dashboards:name=ev-sales-mlops"
    )
    logger.info(f"Monitoring setup complete. Dashboard: {dashboard_url}")


if __name__ == "__main__":
    main()
