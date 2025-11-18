from datetime import datetime, timedelta
import os
import uuid
from dataclasses import dataclass
from temporalio import activity, workflow, client
import random
import polars as pl
from model_training.model_training_workflow import (
    ModelTrainingWorkflow,
    ModelTrainingInput,
)
from model_inferencing.model_inference_workflow import ModelInferenceWorkflow
from model_inferencing.model_inference_activities import InferenceParams


@activity.defn
async def hello_world(name: str) -> str:
    return f"Hello, {name}!"


@dataclass
class IngestDataParams:
    num_records: int = 50
    # correlation with creating a deployment
    deployment_correlation: float = 0.65


@activity.defn
def ingest_data(ingest_data_params: IngestDataParams) -> str:
    """
    Generate synthetic data for trial conversion prediction.

    :return: filepath containing latest data
    :rtype: str
    """
    # In prod would use temporal search params to get date range associated with workflow execution
    start_date = (datetime.now() - timedelta(days=7)).date()
    end_date = datetime.now().date() + timedelta(days=7)
    num_records = ingest_data_params.num_records
    correlation = ingest_data_params.deployment_correlation
    file_path = f"data/trial_conversion_data_{datetime.now().date()}.parquet"
    os.makedirs("data", exist_ok=True)
    if os.path.exists(file_path):
        print("Data has already been ingested for the specified date range")
        return file_path

    # Generate random dates within the specified range
    def random_date(start, end):
        delta = end - start
        random_days = random.randint(0, delta.days)
        return start + timedelta(days=random_days)

    # Create empty dataframe
    data = []

    # Generate correlated boolean features
    annual_contract_base_rate = 0.4

    successful_jobs_correlation = (
        correlation if correlation else (random.randint(1, 8) / 10)
    )

    deployment_correlation = correlation if correlation else (random.randint(1, 7) / 10)

    print("successful_tasks_correlation:", successful_jobs_correlation)
    print("deployment_correlation:", deployment_correlation)

    # Generate data with correlations
    for i in range(num_records):
        # Generate trial dates (no more than 30 days apart)
        trial_start_date = random_date(start_date, end_date)
        trial_length = random.randint(1, 30)  # 1-30 days
        trial_end_date = trial_start_date + timedelta(days=trial_length)

        # Determine engagement status (40% chance of going annual)
        annual_contract = random.random() < annual_contract_base_rate

        # Generate correlated features
        # For successful jobs, we want correlation of 0.8 with annual_contract
        if annual_contract:
            has_successful_jobs = random.random() < (
                0.5 + successful_jobs_correlation / 2
            )
            is_first_deployment = random.random() < (0.5 + deployment_correlation / 2)
        else:
            has_successful_jobs = random.random() < (
                0.5 - successful_jobs_correlation / 2
            )
            is_first_deployment = random.random() < (0.5 - deployment_correlation / 2)

        # Other features with some degree of correlation, but less controlled
        if annual_contract:
            is_business_email = random.random() < 0.3
            is_payment_method_added = random.random() < 0.6
        else:
            is_business_email = random.random() < 0.4
            is_payment_method_added = random.random() < 0.2

        # Generate a record
        record = {
            "ORG_ID": str(uuid.uuid4())[:8],  # Generate a short unique ID
            "ACCT_ID": str(uuid.uuid4())[:10],
            "ACCT_NAME": f"Test Account {i + 1}",
            "TRIAL_START_DATE": trial_start_date.strftime("%Y-%m-%d"),
            "TRIAL_END_DATE": trial_end_date.strftime("%Y-%m-%d"),
            "IS_BUSINESS_EMAIL": is_business_email,
            "IS_FIRST_DEPLOYMENT_CREATED_WITHIN_1D": is_first_deployment,
            "HAS_SUCCESFUL_JOBS_WITHIN_1D": has_successful_jobs,
            "IS_PAYMENT_METHOD_ADDED": is_payment_method_added,
            "ANNUAL_CONTRACT": annual_contract,
        }

        data.append(record)

    # Convert to DataFrame
    df = pl.DataFrame(data)

    df.write_parquet(file_path)
    return file_path


@activity.defn
async def train_model_if_not_exists():
    if not os.path.exists("models/champion/champion_model.joblib"):
        temporal_client = await client.Client.connect("localhost:7233")
        model_output = await temporal_client.execute_workflow(
            ModelTrainingWorkflow.run,
            ModelTrainingInput(),
            id=f"model-training-workflow-{uuid.uuid4()}",
            task_queue="default-queue",
        )
        return model_output
    else:
        print("champion model already exists, skipping training")
        return


@activity.defn
async def run_predictions_workflow():
    temporal_client = await client.Client.connect("localhost:7233")
    return await temporal_client.execute_workflow(
        ModelInferenceWorkflow.run,
        InferenceParams(),
        id=f"model-inference-workflow-{uuid.uuid4()}",
        task_queue="default-queue",
    )
