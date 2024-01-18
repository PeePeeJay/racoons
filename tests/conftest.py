import pytest
from sklearn.datasets import make_classification

import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture()
def forest_based_model():
    pass


@pytest.fixture()
def classification_data():
    X, y = make_classification(
        n_samples=100,
        n_features=30,
        n_redundant=10,
        n_classes=2,
    )
    df = pd.DataFrame(X).add_prefix("feature_")
    df.insert(0, "outcome", y)

    # cast to correct dtype and add random np.nan values

    df['outcome'] = df['outcome'].astype(bool)
    df['feature_0'] = df['feature_0'].astype(int)
    df['feature_0'] = df['feature_0'].astype(pd.Int64Dtype())
    df.loc[0, 'feature_0'] = np.nan
    df['feature_1'] = np.random.randint(0, 4, size=df.shape[0])
    df.loc[0, 'feature_1'] = np.nan
    df['feature_1'] = df["feature_1"].astype(pd.CategoricalDtype())
    df.loc[3, 'feature_3'] = np.nan

    # target and feature columns
    target_col = ["outcome"]
    feature_cols = df.columns.tolist()
    feature_cols.remove("outcome")

    return df, target_col, feature_cols


@pytest.fixture()
def output_path():
    return Path("default")
