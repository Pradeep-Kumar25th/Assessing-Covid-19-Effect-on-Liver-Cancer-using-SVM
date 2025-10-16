from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.data import load_dataset, train_test_split_stratified
from ml_pipeline.preprocess import build_preprocessor
from ml_pipeline.models import build_model_grids, build_search
from ml_pipeline.evaluate import evaluate_classifier, plot_roc_curve


DEFAULT_TARGET = "Alive_Dead"


def main():
    parser = argparse.ArgumentParser(description="Train multiple classifiers with preprocessing and CV")
    parser.add_argument("--csv", type=str, default="covid-liver.csv")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="f1")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.csv, args.target)

    # Normalize target to binary 0/1
    y_normalized = data.y.astype(str).str.lower().map({"dead": 1, "alive": 0})
    if y_normalized.isna().any():
        # If mapping fails, try to infer numeric binary
        try:
            y_normalized = pd.to_numeric(data.y)
        except Exception as e:
            raise ValueError("Target must be binary (Alive/Dead or 0/1)") from e

    X_train, X_test, y_train, y_test = train_test_split_stratified(data.X, y_normalized)

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    models = build_model_grids(preprocessor)

    best_models = {}
    results_summary = {}

    for name, (pipeline, grid) in models.items():
        print(f"\nTraining model: {name}")
        search = build_search(pipeline, grid, scoring=args.scoring, cv=args.cv, n_jobs=args.n_jobs)
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_

        # Save CV results
        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_path = outdir / f"{name}_cv_results.csv"
        cv_results_df.to_csv(cv_results_path, index=False)

        # Evaluate on test set
        eval_res = evaluate_classifier(search.best_estimator_, X_test, y_test)
        results_summary[name] = {
            "best_params": search.best_params_,
            "metrics": {
                "accuracy": eval_res.accuracy,
                "precision": eval_res.precision,
                "recall": eval_res.recall,
                "f1": eval_res.f1,
                "roc_auc": eval_res.roc_auc,
            },
        }

        # Save ROC plot
        roc_path = outdir / f"{name}_roc.png"
        plot_roc_curve(search.best_estimator_, X_test, y_test, out_path=str(roc_path))

        # Save the best model
        model_path = outdir / f"{name}_best_model.joblib"
        joblib.dump(search.best_estimator_, model_path)

    # Save summary JSON
    summary_path = outdir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\nTraining complete. Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
