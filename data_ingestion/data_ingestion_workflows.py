from dataclasses import dataclass
from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from data_ingestion.data_ingestion_activities import (
        hello_world,
        IngestDataParams,
        ingest_data,
        train_model_if_not_exists,
        run_predictions_workflow,
    )


@dataclass
class DataIngestionOutput:
    data: str
    model: str | None = None
    predictions: str | None = None


@workflow.defn
class HelloWorldWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            hello_world,
            name,
            schedule_to_close_timeout=timedelta(seconds=10),
        )


@workflow.defn
class DataIngestionWorkflow:
    @workflow.run
    async def run(self, params: IngestDataParams | None = None) -> str:
        # Implementation for data ingestion using params
        data = await workflow.execute_activity(
            ingest_data,
            params if params else IngestDataParams(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )

        model = await workflow.execute_activity(
            train_model_if_not_exists, schedule_to_close_timeout=timedelta(minutes=15)
        )

        predictions = await workflow.execute_activity(
            run_predictions_workflow,
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        output = {"data": data, "predictions": predictions}
        if model:
            output["model"] = model

        return DataIngestionOutput(**output)
