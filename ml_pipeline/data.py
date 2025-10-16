from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataSplit:
    X: pd.DataFrame
    y: pd.Series


def load_dataset(csv_path: str, target_column: str) -> DataSplit:
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not in dataset. Available: {list(df.columns)}")

    y = df[target_column]
    X = df.drop(columns=[target_column])
    return DataSplit(X=X, y=y)


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
