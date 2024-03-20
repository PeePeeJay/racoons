import numpy as np
import pandas as pd
from typing import List
from pandas.core.dtypes.common import is_bool_dtype, is_float_dtype
from collections import defaultdict
from pandas.core.dtypes.common import is_bool_dtype, is_float_dtype, is_integer_dtype
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from racoons.models import supported_scale_levels


def features_and_targets_from_dataframe(
    df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Split features from targets in a given dataframe and determine the scale level of features.

    Args:
        df (pd.DataFrame): The dataframe containing target columns and feature columns.
        feature_cols (list[str]): Columns containing features.
        target_cols (list[str]): Columns containing targets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: A tuple containing features, targets, and feature scale levels.

    Example:
        >>> df = pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]})
        >>> features, targets, feature_scale_levels = features_and_targets_from_dataframe(df, ['feature1'], ['target'])

    Note:
        This function checks the scale level of feature columns and ensures valid target columns.
    """
    # check scale level of feature columns
    feature_cols_selected = []
    numerical_features, ordinal_features, categorical_features = [], [], []

    for col in feature_cols:
        scale_level = get_scale_level(df[col])
        if scale_level in supported_scale_levels:
            feature_cols_selected.append(col)
            if scale_level == "numerical":
                numerical_features.append(col)
            elif scale_level == "ordinal":
                ordinal_features.append(col)
            elif scale_level == "categorical":
                categorical_features.append(col)
        else:
            print(
                f"Feature column {col} has an unsupported dtype {df[col].dtype} and will be dropped.\n"
            )
    print(
        f"{len(feature_cols_selected)} features out of {len(feature_cols)} "
        f"initial features were selected for analysis.\n"
        f"numerical features: {len(numerical_features)}\n"
        f"ordinal features: {len(ordinal_features)}\n"
        f"categorical features: {len(categorical_features)}\n"
    )
    # check if target columns are binary or numeric if multiclass
    target_cols_selected = []
    for col in target_cols:
        if is_bool_dtype(df[col].dtype) or is_integer_dtype(df[col].dtype):
            target_cols_selected.append(col)
        else:
            print(
                f"Target column {col} has an unsupported dtype {df[col].dtype} and will be dropped.\n"
            )
    print(
        f"{len(target_cols_selected)} targets out of {len(target_cols)} "
        f"initial targets were selected for analysis.\n"
    )

    df = df.loc[:, target_cols_selected + feature_cols_selected]

    feature_scale_levels = {
        "numerical": numerical_features,
        "ordinal": ordinal_features,
        "categorical": categorical_features,
    }

    return (
        df.loc[:, feature_cols_selected],
        df.loc[:, target_cols_selected],
        feature_scale_levels
    )


def get_scale_level(feature: pd.Series) -> str:
    """
    Determine the scale level of a feature based on its data type.

    Args:
        feature (pd.Series): The Pandas Series representing the feature.

    Returns:
        str: The scale level of the feature, which can be one of the following:
            - 'numerical': If the feature has a data type of float.
            - 'ordinal': If the feature has a data type of pd.Int64Dtype.
            - 'categorical': If the feature has a data type of pd.CategoricalDtype.
            - None: If the feature has an unsupported data type.

    Raises:
        None

    Example:
        >>> import pandas as pd
        >>> feature = pd.Series([1, 2, 3], dtype=float)
        >>> get_scale_level(feature)
        'numerical'

    Note:
        This function is designed to determine the scale level of a feature based on its data type.
        If the data type is not supported, it returns None.
    """
    if is_float_dtype(feature.dtype):
        return "numerical"
    elif feature.dtype == np.int64:
        return "ordinal"
    elif isinstance(feature.dtype, pd.CategoricalDtype):
        return "categorical"
    else:
        print(
            f"The feature {feature.name} has an unsupported dtype '{feature.dtype}' and will be dropped."
        )


def create_scale_level_template(df: pd.DataFrame, columns_to_use: List[str] = None) -> pd.DataFrame:
    """Creates a table to be filled with the correct scale levels"""
    dtypes = []
    levels = []
    if not columns_to_use:
        columns_to_use = df.columns.tolist()
    for column in df.columns:
        if column in columns_to_use:
            if pd.api.types.is_string_dtype(df[column].dtype) or pd.api.types.is_object_dtype(df[column].dtype) or pd.api.types.is_bool_dtype(df[column].dtype):
                dtypes.append("categorical")
                levels.append(None)
            elif pd.api.types.is_numeric_dtype(df[column].dtype):
                dtypes.append("numerical")
                levels.append(None)
            else:
                dtypes.append("provide scale level")
                levels.append(None)

    report_df = pd.DataFrame({"Column": columns_to_use, "Scale Level": dtypes, "Level order (for ordinal values)": levels})
    return report_df


def encode_multitarget_data(df: pd.DataFrame, target_columns: list[str]) -> tuple[pd.DataFrame, defaultdict]:
    label_encoders = defaultdict(LabelEncoder)
    for col in target_columns:
        df[col] = label_encoders[col].fit_transform(df[col])
    return df, label_encoders
