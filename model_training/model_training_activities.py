from datetime import datetime
from enum import Enum
import os
import shutil
from dataclasses import dataclass, field
from temporalio import activity
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
)
import joblib


@dataclass
class ModelTrainingConfig:
    model: str
    hyperparameters: dict


@dataclass
class ModelTrainingParams:
    model_training_params: list[ModelTrainingConfig] = field(
        default_factory=lambda: [
            ModelTrainingConfig(
                "LogisticRegression",
                {
                    "C": [0.1, 1, 10, 100],
                    "penalty": ["l1", "l2", "elasticnet"],
                    "solver": ["liblinear", "saga"],
                },
            ),
            ModelTrainingConfig(
                "RandomForest",
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                },
            ),
        ]
    )


@dataclass
class EvaluationMetric:
    metric: str = "f1_score"


@dataclass
class ModelOutput:
    model: str
    model_training_params: dict
    accuracy: float
    filepath: str
    roc_auc: float | None
    f1_score: float


@dataclass
class ChampionModel(ModelOutput):
    champion_path: str


@activity.defn
def train_model(
    file_path: str, model_training_config: ModelTrainingConfig
) -> ModelOutput:
    """
    Trains model based on the provided configuration
    """

    df = pl.read_parquet(file_path, glob=True)
    df = df.select(
        pl.col("IS_BUSINESS_EMAIL").cast(pl.Boolean),
        pl.col("IS_FIRST_DEPLOYMENT_CREATED_WITHIN_1D").cast(pl.Boolean),
        pl.col("HAS_SUCCESFUL_JOBS_WITHIN_1D").cast(pl.Boolean),
        pl.col("IS_PAYMENT_METHOD_ADDED").cast(pl.Boolean),
        pl.col("ANNUAL_CONTRACT").cast(pl.Int64),
    )
    supported_models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
    }

    model = supported_models.get(model_training_config.model)

    if model is None:
        raise ValueError(f"Unsupported model type: {model_training_config.model}")

    X = df.select(
        pl.col("IS_BUSINESS_EMAIL"),
        pl.col("IS_FIRST_DEPLOYMENT_CREATED_WITHIN_1D"),
        pl.col("HAS_SUCCESFUL_JOBS_WITHIN_1D"),
        pl.col("IS_PAYMENT_METHOD_ADDED"),
    )
    y = df.select(pl.col("ANNUAL_CONTRACT"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=49
    )

    grid_search = RandomizedSearchCV(
        model,
        model_training_config.hyperparameters,
        cv=3,  # Fewer folds
        scoring="average_precision",
        n_iter=10,  # Randomized search
        n_jobs=-1,  # Use all CPU cores
        random_state=23,
    )
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    y_proba = (
        grid_search.predict_proba(X_test)[:, 1]
        if hasattr(grid_search, "predict_proba")
        else None
    )

    # get real distribution of classes
    # target_dist = y_test.to_series(0).value_counts(normalize=True)

    print(f"Grid search over {model_training_config.model} complete.")
    print(f"Best params: {grid_search.best_params_}")
    print("Scoring...")
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    # classification_rep = classification_report(y_test, y_pred)
    os.makedirs("models/candidates", exist_ok=True)
    filepath = (
        f"models/candidates/{model_training_config.model}_{datetime.now()}.joblib"
    )
    joblib.dump(grid_search.best_estimator_, filepath)
    tuned_model = {
        "model": model_training_config.model,
        "model_training_params": grid_search.best_params_,
        "accuracy": float(round(acc, 5)),
        "filepath": filepath,
        # "classification_report": classification_rep,
        "roc_auc": float(round(roc_auc, 5)),
        "f1_score": float(round(f1, 5)),
        # "model_training_target_distribution": float(round(target_dist.to_dict()[1], 5)),
    }
    return ModelOutput(**tuned_model)


@activity.defn
def choose_best_model(
    evaluation_metric: EvaluationMetric, models: list[ModelOutput]
) -> ChampionModel:
    """Chooses the best model based on the evaluation metric"""

    if evaluation_metric.metric in ["accuracy", "f1_score", "roc_auc"]:
        best_model = max(models, key=lambda m: m.__dict__[evaluation_metric.metric])
    else:
        raise ValueError(f"Unknown selection metric: {evaluation_metric}")

    model_path = best_model.filepath
    os.makedirs("models/champion", exist_ok=True)
    champion_path = "models/champion/champion_model.joblib"
    champion_hash = "models/champion/champion_hash.txt"
    current_hash = str(
        hash(
            (
                best_model.model,
                str(best_model.model_training_params),
            )
        )
    )
    # Use a deterministic champion path based on model characteristics
    if os.path.exists(champion_hash):
        with open(champion_hash, "r") as f:
            existing_hash = f.read().strip()

        if existing_hash == current_hash:
            print("Champion model is already up to date.")
            return ChampionModel(**best_model.__dict__, champion_path=champion_path)
    else:
        with open(champion_hash, "w") as f:
            f.write(current_hash)

    # Overwrite existing champion model
    shutil.copy(model_path, champion_path)
    print(f"New champion model saved to: {champion_path}")

    return ChampionModel(**best_model.__dict__, champion_path=champion_path)
