"""Trains a small classifier, logs it to a local MLflow model registry, and
saves a reference feature distribution (used later for drift detection at
serving time). Run this once before starting the API.
"""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
FEATURE_NAMES = [f"feature_{i}" for i in range(5)]


def train_and_register(n_samples: int = 2000, random_state: int = 42) -> str:
    X, y = make_classification(
        n_samples=n_samples, n_features=5, n_informative=3, n_redundant=1,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    mlflow.set_experiment("model-serving-platform")

    with mlflow.start_run(run_name="random_forest_v1") as run:
        params = {"n_estimators": 100, "max_depth": 6, "random_state": random_state}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, artifact_path="model", serialization_format="pickle",
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        # Reference distribution for drift detection: the training feature
        # values themselves, so serving-time incoming data can be compared
        # against what the model actually saw during training.
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        reference = {
            "feature_names": FEATURE_NAMES,
            "reference_samples": X_train.tolist(),
        }
        with open(ARTIFACTS_DIR / "reference_distribution.json", "w") as f:
            json.dump(reference, f)

        with open(ARTIFACTS_DIR / "model_uri.txt", "w") as f:
            f.write(model_uri)

        print(f"Trained model. run_id={run_id}")
        print(f"Metrics: {metrics}")
        print(f"Model URI: {model_uri}")

        return model_uri


if __name__ == "__main__":
    train_and_register()
