import pandas
import pandas as pd
from pandas.core.dtypes.common import is_bool_dtype, is_float_dtype

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
    # check if target columns are binary
    target_cols_selected = []
    for col in target_cols:
        if is_bool_dtype(df[col].dtype):
            target_cols_selected.append(col)
        else:
            print(
                f"Target column {col} has an unsupported dtype {df[col].dtype} and will be dropped.\n"
            )
    print(
        f"{len(target_cols_selected)} targets out of {len(target_cols)} "
        f"initial targets were selected for analysis.\n"
    )

    feature_scale_levels = {
        "numerical": numerical_features,
        "ordinal": ordinal_features,
        "categorical": categorical_features,
    }
    return (
        df.loc[:, feature_cols_selected],
        df.loc[:, target_cols_selected],
        feature_scale_levels,
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
    elif isinstance(feature.dtype, pd.Int64Dtype):
        return "ordinal"
    elif isinstance(feature.dtype, pd.CategoricalDtype):
        return "categorical"
    else:
        print(
            f"The feature {feature.name} has an unsupported dtype '{feature.dtype}' and will be dropped."
        )
