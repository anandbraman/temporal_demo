import asyncio
import argparse
import uuid
from temporalio.client import Client


async def main():
    parser = argparse.ArgumentParser(description="Start Model Training Workflow")
    parser.add_argument("--workflow_name", required=True)
    parser.add_argument("--workflow_id", default=None)
    args = parser.parse_args()

    if not args.workflow_id:
        args.workflow_id = f"{args.workflow_name}-{uuid.uuid4()}"
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        args.workflow_name,
        id=args.workflow_id,
        task_queue="default-queue",
    )
    print("Workflow result:", result)


if __name__ == "__main__":
    asyncio.run(main())
