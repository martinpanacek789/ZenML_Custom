import numpy as np
from zenml import step, pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

@step
def load_data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

@step
def train_model_rfc(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    return rfc, y_pred, rfc.score(X_test, y_test)

@step
def train_model_gbc(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import GradientBoostingClassifier

    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    return gbc, y_pred, gbc.score(X_test, y_test)

@pipeline(enable_cache=False)
def training_pipeline_local():
    X_train, X_test, y_train, y_test = load_data()
    model_rfc, y_pred_rfc, _ = train_model_rfc(X_train, X_test, y_train, y_test)
    model_gbc, y_pred_gbc, _ = train_model_gbc(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    training_pipeline_local()
