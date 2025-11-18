import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker
from data_ingestion.data_ingestion_activities import (
    hello_world,
    ingest_data,
    run_predictions_workflow,
    train_model_if_not_exists,
)
from data_ingestion.data_ingestion_workflows import (
    HelloWorldWorkflow,
    DataIngestionWorkflow,
)
from model_inferencing.model_inference_workflow import ModelInferenceWorkflow
from model_inferencing.model_inference_activities import make_predictions, send_outreach
from model_training.model_training_workflow import ModelTrainingWorkflow
from model_training.model_training_activities import train_model, choose_best_model


async def main():
    with ThreadPoolExecutor(max_workers=10) as executor:
        client = await Client.connect("localhost:7233")
        worker = Worker(
            client,
            task_queue="default-queue",
            activity_executor=executor,
            workflows=[
                HelloWorldWorkflow,
                DataIngestionWorkflow,
                ModelTrainingWorkflow,
                ModelInferenceWorkflow,
            ],
            activities=[
                hello_world,
                ingest_data,
                train_model,
                choose_best_model,
                make_predictions,
                train_model_if_not_exists,
                run_predictions_workflow,
                send_outreach,
            ],
        )
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
