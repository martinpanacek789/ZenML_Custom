import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from zenml import pipeline, step
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
from zenml.config import DockerSettings

docker_settings = DockerSettings(requirements="requirements.txt")

@step(settings={"docker": docker_settings}, experiment_tracker="mlflow_experiment_tracker")
def train_model_rfc() -> BaseEstimator:
    mlflow.autolog()
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model_rfc = RandomForestClassifier()
    model_rfc.fit(X_train, y_train)
    y_pred = model_rfc.predict(X_test)
    mlflow.log_param("n_estimators_rf", model_rfc.n_estimators)
    mlflow.log_metric("train_accuracy_rf", accuracy_score(y_test, y_pred))
    return model_rfc


@step(settings={"docker": docker_settings}, experiment_tracker="mlflow_experiment_tracker")
def train_model_gbc() -> BaseEstimator:
    mlflow.autolog()
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model_gbc = GradientBoostingClassifier()
    model_gbc.fit(X_train, y_train)
    y_pred = model_gbc.predict(X_test)
    mlflow.log_param("n_estimators_gb", model_gbc.n_estimators)
    mlflow.log_metric("train_accuracy_gb", accuracy_score(y_test, y_pred))
    return model_gbc

@pipeline(enable_cache=False)
def training_pipeline_tracked():
    train_model_gbc()
    train_model_rfc()

training_pipeline_tracked()
