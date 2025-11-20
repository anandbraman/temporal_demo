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
The `IngestDataWorkflow` is the main workflow. It generates synthetic data, trains models if no champion model exists, and runs predictions. The synthetic data is metadata about a user's trial account for a software. The outcome of interest is whether or not the user will convert from trial to an annual subscription. Because it's synthetic, I generated the labels for the data, but in the real world the active trials would be unlabeled. The ingest data job is scheduled to run daily. To create the schedule, run `python ingest_data_schedule.py`. 

#### Ingest Data Inputs
The inputs are defined by the `IngestDataParams` class. Defaults are set for all vars, but a correlation between the deployment variable and the trial conversion can be set manually. The number of records written to the file can also be configured. 

#### Ingest Data Output
The trial data is output to a file called `data/trial_conversion_data_{run_date}.parquet`. The Ingest Data Job also returns the filepath of the champion model, if training was required (i.e., first run). The location of the prediction file and the orgs + contact emails are also returned. 

### Train Models
The `ModelTrainingWorkflow` trains classification models **in parallel** based on a default config or user provided config. Currently Logistic Regression and Random Forests are the supported classifiers. The example is simple for now, where the best model from the workflow execution is promoted to champion and used in the `ModelInferenceWorkflow`. Evaluation metrics are configurable but the default is f1 score. The model training workflow is scheduled to run monthly. To create the schedule run `python model_training_schedule.py`. Candidate models are saved at `models/candidates/{model_training_config.model}_{datetime.now()}.joblib`. The champion model is written to `models/champion/champion_model.joblib`. The hyperparams of the best model are hashed and saved to `models/champion/champion_hash.txt`

#### Model Training Input
Model training input is defined by the [ModelTrainingInput](https://github.com/anandbraman/temporal_demo/blob/9737ec3121363fcd0345cb8b3131304b8464b774/model_training/model_training_workflow.py#L17) dataclass. This input allows users to select what models are to be trained, configure hyperparameter search values, and set the preferred metric for selecting the champion model (defaults to f1 score).

#### Model Training Output
The details of the champion model are returned by the ModelTrainigWorflow. Output is defined by the [ChampionModel](https://github.com/anandbraman/temporal_demo/blob/9737ec3121363fcd0345cb8b3131304b8464b774/model_training/model_training_activities.py#L66) class.

### Run Inference
The `ModelInferenceWorkflow` makes predictions about which accounts will likely convert from trial to an annual subscription and sends outreach to the main point of contact for the trial account. The threshold for classificaion is parameterizable. Prediction output is saved to `data/inference_results_{datetime.now().date()}.csv`

#### Inference Input
Inference input is defined by the [InferenceParams](https://github.com/anandbraman/temporal_demo/blob/9737ec3121363fcd0345cb8b3131304b8464b774/model_inferencing/model_inference_activities.py#L11C7-L11C22) class. Users can configure the glob for the data, the model used, and the threshhold for classifcation.

#### Inference Output
The Inference Workflow returns [OutreachOuput](https://github.com/anandbraman/temporal_demo/blob/9737ec3121363fcd0345cb8b3131304b8464b774/model_inferencing/model_inference_activities.py#L30). The outreach output contains the location of the predictions file and a list of orgs & contacts at those orgs (faker lib)

## Future Improvements
Future workflows could include an agentic workflow that replies to customers who respond to the outreach. An example could be a workflow that sends pricing resources if a user requests it, or links to docs if a user replies with technical questions. The workflow could also defer to a human in the loop. 
