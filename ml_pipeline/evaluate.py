from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


@dataclass
class EvalResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def evaluate_classifier(model, X_test, y_test) -> EvalResults:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback for models without predict_proba (e.g., SVC with probability=False)
        try:
            y_score = model.decision_function(X_test)
            # Min-max scale decision function to [0,1]
            y_proba = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)
        except Exception:
            y_proba = None

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    if y_proba is not None:
        roc_auc = metrics.roc_auc_score(y_test, y_proba)
    else:
        roc_auc = float("nan")

    return EvalResults(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
    )


def plot_roc_curve(model, X_test, y_test, out_path: str | None = None):
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_score = model.decision_function(X_test)
            y_proba = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)
        except Exception:
            return

    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()
