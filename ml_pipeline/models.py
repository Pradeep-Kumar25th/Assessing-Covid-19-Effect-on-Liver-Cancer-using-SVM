from __future__ import annotations

from typing import Dict, Tuple

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def build_model_grids(preprocessor) -> Dict[str, Tuple[Pipeline, Dict[str, list]]]:
    models = {}

    # Logistic Regression
    log_reg = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None))
    ])
    log_reg_grid = {
        "clf__penalty": ["l2"],
        "clf__C": [0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs", "liblinear"],
    }
    models["log_reg"] = (log_reg, log_reg_grid)

    # Support Vector Machine
    svm = Pipeline([
        ("prep", preprocessor),
        ("clf", SVC(probability=True))
    ])
    svm_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    }
    models["svm"] = (svm, svm_grid)

    # Random Forest
    rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    rf_grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    }
    models["rf"] = (rf, rf_grid)

    # Gradient Boosting
    gb = Pipeline([
        ("prep", preprocessor),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])
    gb_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
    }
    models["gb"] = (gb, gb_grid)

    return models


def build_search(model: Pipeline, param_grid: Dict[str, list], scoring="f1", cv: int = 5, n_jobs: int = -1) -> GridSearchCV:
    scorer = make_scorer(f1_score, average="binary") if scoring == "f1" else scoring
    return GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
