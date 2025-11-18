# Predicting Trial to Annual Conversion with Temporal

## Setup
Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```
# Create your virutal environment:
uv venv .venv --python 3.13

# Install the requirements: 
uv pip install -r requirements.txt

# run the worker
source .venv/bin/activate
python worker.py

# in a new terminal
source .venv/bin/activate
python starter.py --workflow_name DataIngestionWorkflow
```

## Workflows
1. Ingest Data
2. Train Model
3. Make Predictions

### Ingest Data
The `IngestDataWorkflow` is the main workflow. It generates synthetic data, trains models if no champion model exists, and runs predictions. The synthetic data is metadata about a user's trial account for a software. The outcome of interest is whether or not the user will convert from trial to an annual subscription. Because it's synthetic, I generated the labels for the data, but in the real world the active trials would be unlabeled. The ingest data job is scheduled to run daily. To create the schedule, run `python ingest_data_schedule.py`

### Train Models
The `ModelTrainingWorkflow` trains classification models **in parallel** based on a default config or user provided config. Currently Logistic Regression and Random Forests are the supported classifiers. The example is simple for now, where the best model from the workflow execution is promoted to champion and used in the `ModelInferenceWorkflow`. Evaluation metrics are configurable but the default is f1 score. The model training workflow is scheduled to run monthly. To create the schedule run `python model_training_schedule.py`

### Run Inference
The `ModelInferenceWorkflow` makes predictions about which accounts will likely convert from trial to an annual subscription and sends outreach to the main point of contact for the trial account. The threshold for classificaion is parameterizable. 

## Future Improvements
Future workflows could include an agentic workflow that replies to customers who respond to the outreach. An example could be a workflow that sends pricing resources if a user requests it, or links to docs if a user replies with technical questions. The workflow could also defer to a human in the loop. 