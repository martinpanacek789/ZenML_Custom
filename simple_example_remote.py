import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from zenml import pipeline, step
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from zenml.config import DockerSettings


docker_settings = DockerSettings(requirements="requirements.txt")

@step(settings={"docker": docker_settings})
def train_model_rfc() -> tuple[BaseEstimator, np.ndarray, float]:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model_rfc = RandomForestClassifier()
    model_rfc.fit(X_train, y_train)
    y_pred_rfc = model_rfc.predict(X_test)
    return model_rfc, y_pred_rfc, accuracy_score(y_test, y_pred_rfc)


@step(settings={"docker": docker_settings})
def train_model_gbc() -> tuple[BaseEstimator, np.ndarray, float]:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model_gbc = GradientBoostingClassifier()
    model_gbc.fit(X_train, y_train)
    y_pred_gbc = model_gbc.predict(X_test)
    return model_gbc, y_pred_gbc, accuracy_score(y_test, y_pred_gbc)


def evaluate_ensemble(y_pred_rfc, y_pred_gbc, y_test):
    return accuracy_score(y_test, (y_pred_rfc + y_pred_gbc) / 2)

@pipeline(enable_cache=False)
def training_pipeline_remote():
    model_rfc, y_pred_rfc, _ = train_model_rfc()
    model_gbc, y_pred_gbc, _ = train_model_gbc()
    ensemble_accuracy = evaluate_ensemble(y_pred_rfc, y_pred_gbc, y_test)

training_pipeline_remote()
