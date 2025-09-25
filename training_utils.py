
from typing import Any, Dict
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def make_default_pipelines(random_state=42):

    pipelines = {}

    # SVM linear
    pipelines["SVM_linear"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", random_state=random_state))
    ])

    # SVM rbf
    pipelines["SVM_rbf"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", random_state=random_state))
    ])

    # Logistic (linear)
    pipelines["Logistic_linear"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
    ])
    return pipelines

def train_classifier_cv(X, y, cv: int=5, random_state: int=42) -> Dict[str, Any]:

    labels = np.unique(y)
    pipelines = make_default_pipelines(random_state=random_state)
    results = {}

    for name, pipe in pipelines.items():
        # cross_validation accuracy
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()

        y_pred = cross_val_predict(pipe, X, y, cv=cv, method='predict')
        cm = confusion_matrix(y, y_pred, labels=labels)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True))

        results[name] = {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "confusion_matrix": cm,
            "confusion_matrix_norm": cm_norm
        }
    return results


