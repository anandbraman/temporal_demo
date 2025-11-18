from temporalio import workflow
from datetime import timedelta
from dataclasses import dataclass, field
import asyncio


with workflow.unsafe.imports_passed_through():
    from model_training.model_training_activities import (
        ModelTrainingParams,
        EvaluationMetric,
        train_model,
        choose_best_model,
    )


@dataclass
class ModelTrainingInput:
    file_path: str | None = "data/*.parquet"
    evaluation_metric: EvaluationMetric = field(
        default_factory=lambda: EvaluationMetric()
    )
    model_training_params: ModelTrainingParams = field(
        default_factory=lambda: ModelTrainingParams()
    )


@workflow.defn
class ModelTrainingWorkflow:
    @workflow.run
    async def run(self, input: ModelTrainingInput | None = None) -> dict:
        input = input if input else ModelTrainingInput()

        # Create coroutines for all model training tasks
        training_tasks = []
        for model_config in input.model_training_params.model_training_params:
            task = workflow.execute_activity(
                train_model,
                args=[
                    input.file_path,
                    model_config,
                ],
                schedule_to_close_timeout=timedelta(minutes=10),
            )
            training_tasks.append(task)

        # Execute all training tasks in parallel
        model_candidates = await asyncio.gather(*training_tasks)

        best_model = await workflow.execute_activity(
            choose_best_model,
            args=[
                input.evaluation_metric,
                model_candidates,
            ],
            schedule_to_close_timeout=timedelta(minutes=2),
        )
        # define a dataclass for output
        return best_model
