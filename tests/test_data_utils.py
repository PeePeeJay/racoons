import pytest
import pandas as pd
from racoons.data_utils import features_and_targets_from_dataframe, get_scale_level
from sklearn.model_selection import cross_validate


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (pd.Series([1, 2, 3], dtype=float), "numerical"),
        (pd.Series([1, 2, 3], dtype=pd.Int64Dtype()), "ordinal"),
        (pd.Series(["a", "b", "c"], dtype="category"), "categorical"),
        (pd.Series([1, 2, 3], dtype=int), None),  # Unsupported dtype, expecting None
        (
            pd.Series([True, False, True], dtype=bool),
            None,
        ),  # Unsupported dtype, expecting None
    ],
)
def test_get_scale_level(test_input, expected_output):
    result = get_scale_level(test_input)
    assert result == expected_output


def test_features_and_targets_from_dataframe(classification_data):
    df, target_cols, feature_cols = classification_data
    X, y, feature_scale_levels = features_and_targets_from_dataframe(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )

    assert feature_scale_levels["categorical"] == [
        "feature_1",
    ]
    assert feature_scale_levels["numerical"] == [
        f"feature_{i}" for i in range(2, len(feature_cols))
    ]
    assert feature_scale_levels["ordinal"] == ["feature_0"]
    assert all(
        features
        in feature_scale_levels["categorical"]
        + feature_scale_levels["ordinal"]
        + feature_scale_levels["numerical"]
        for features in X.columns.tolist()
    )
    assert y.columns.tolist() == target_cols
