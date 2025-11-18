"""
Workflow that makes predictions on new data. Invoked by the data ingestion workflow

Inference workflow filters for active trials, makes a prediction,
and sends outreach to orgs likely to convert.
"""

from temporalio import workflow
from datetime import timedelta

with workflow.unsafe.imports_passed_through():
    from model_inferencing.model_inference_activities import (
        InferenceParams,
        make_predictions,
        send_outreach,
        OutreachOutput,
    )


@workflow.defn
class ModelInferenceWorkflow:
    @workflow.run
    async def run(self, input: InferenceParams | None = None) -> OutreachOutput:
        input = input if input else InferenceParams()

        prediction_output = await workflow.execute_activity(
            make_predictions,
            args=[input],
            schedule_to_close_timeout=timedelta(minutes=5),
        )

        outreach_output = await workflow.execute_activity(
            send_outreach,
            args=[prediction_output],
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        return outreach_output
