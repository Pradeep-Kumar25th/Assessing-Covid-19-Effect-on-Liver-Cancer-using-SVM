from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_MODEL = "artifacts/svm_best_model.joblib"


def main():
    parser = argparse.ArgumentParser(description="Run inference using a saved model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input_csv)

    # If input contains the target, drop it
    for tgt in ["Alive_Dead", "target", "label"]:
        if tgt in df.columns:
            df = df.drop(columns=[tgt])
            break

    preds = model.predict(df)

    # Attempt to map 0/1 back to Alive/Dead
    try:
        pred_labels = pd.Series(preds).map({0: "Alive", 1: "Dead"}).fillna(preds)
    except Exception:
        pred_labels = preds

    out_df = df.copy()
    out_df["prediction"] = pred_labels
    out_path = Path(args.output_csv)
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
