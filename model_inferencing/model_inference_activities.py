from datetime import datetime
from dataclasses import dataclass
import os
from temporalio import activity
import polars as pl
import joblib
from faker import Faker


@dataclass
class InferenceParams:
    data: str = "data/*.parquet"
    model_path: str = "models/champion/champion_model.joblib"
    threshold: float = 0.5


@dataclass
class PredictionOutput:
    predictions_file: str
    orgs_for_outreach: list[str]


@dataclass
class OutreachContact:
    org_id: str
    contact_email: str


@dataclass
class OutreachOutput:
    predictions_file: str
    contacts: list[OutreachContact]


@activity.defn
def make_predictions(inference_params: InferenceParams) -> PredictionOutput:
    """
    Make predictions on active trials
    """
    file = f"data/inference_results_{datetime.now().date()}.csv"
    if os.path.exists(file):
        print("predictions file already exists")
        predictions = pl.read_csv(file)

        return PredictionOutput(
            predictions_file=file,
            orgs_for_outreach=predictions.filter(
                pl.col("PREDICTION_PROBABILITY") > inference_params.threshold
            )
            .select(pl.col("ORG_ID"))
            .to_series()
            .to_list(),
        )

    model = joblib.load(inference_params.model_path)
    data = pl.read_parquet(inference_params.data)

    features = data.select(
        pl.col("IS_BUSINESS_EMAIL").cast(pl.Boolean),
        pl.col("IS_FIRST_DEPLOYMENT_CREATED_WITHIN_1D").cast(pl.Boolean),
        pl.col("HAS_SUCCESFUL_JOBS_WITHIN_1D").cast(pl.Boolean),
        pl.col("IS_PAYMENT_METHOD_ADDED").cast(pl.Boolean),
    )
    predictions = model.predict_proba(features)[:, 1]

    data_with_predictions = data.with_columns(
        pl.Series("PREDICTION_PROBABILITY", predictions)
    )

    data_with_predictions.write_csv(file)

    orgs_for_outreach = data.filter(
        pl.col("PREDICTION_PROBABILITY") > inference_params.threshold
    ).select(pl.col("ORG_ID"))

    return PredictionOutput(
        predictions_file=file, orgs_for_outreach=orgs_for_outreach.to_series().to_list()
    )


@activity.defn
def send_outreach(prediction_output: PredictionOutput) -> OutreachOutput:
    """
    Send outreach to organizations identified for outreach.

    :param orgs: Description
    :type orgs: PredictionOutput
    """
    faker = Faker()
    # send some emails here in prod
    return OutreachOutput(
        predictions_file=prediction_output.predictions_file,
        contacts=[
            OutreachContact(org_id=id, contact_email=faker.email())
            for id in prediction_output.orgs_for_outreach
        ],
    )
