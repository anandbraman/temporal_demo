import asyncio
import uuid
from datetime import timedelta
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
    ScheduleState,
)
from data_ingestion.data_ingestion_workflows import DataIngestionWorkflow


async def main():
    client = await Client.connect("localhost:7233")

    await client.create_schedule(
        "data-ingestion-daily",
        Schedule(
            action=ScheduleActionStartWorkflow(
                DataIngestionWorkflow.run,
                id=f"scheduled-data-ingestion-workflow-{uuid.uuid4()}",
                task_queue="default-queue",
            ),
            spec=ScheduleSpec(
                intervals=[ScheduleIntervalSpec(every=timedelta(days=1))]
            ),
            state=ScheduleState(note="Daily Data Ingestion Job"),
        ),
    )
    print("Created daily data ingestion schedule.")


if __name__ == "__main__":
    asyncio.run(main())
