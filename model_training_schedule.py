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
from model_training.model_training_workflow import ModelTrainingWorkflow


async def main():
    client = await Client.connect("localhost:7233")

    await client.create_schedule(
        "model-training-monthly",
        Schedule(
            action=ScheduleActionStartWorkflow(
                ModelTrainingWorkflow.run,
                id=f"scheduled-model-training-workflow-{uuid.uuid4()}",
                task_queue="default-queue",
            ),
            spec=ScheduleSpec(
                intervals=[ScheduleIntervalSpec(every=timedelta(months=1))]
            ),
            state=ScheduleState(note="Monthly Model Training Job"),
        ),
    )
    print("Created monthly model training schedule.")


if __name__ == "__main__":
    asyncio.run(main())
