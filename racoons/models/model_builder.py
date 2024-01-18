from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from racoons.models import classifiers, sample_methods, feature_selection_methods


def get_estimator(estimator_name: str) -> list[tuple[str, object]]:
    """
    Get an estimator as a pipeline step based on its string representation.

    Args:
        estimator_name (str): String representation of the estimator.

    Returns:
        list[tuple[str, object]]: A pipeline-step-like object to be integrated into the model pipeline.

    Raises:
        NotImplementedError: If the specified estimator is not implemented.

    Example:
        >>> get_estimator('random_forest')
        [('estimator', RandomForestClassifier(n_jobs=-1))]

    Note:
        Available classifiers can be found in the 'classifiers' dictionary.
    """
    if estimator_name not in classifiers.keys():
        raise NotImplementedError(
            f"The classifier '{estimator_name}' is not implemented."
        )
    else:
        return [("estimator", classifiers[estimator_name])]


def get_preprocessing_steps(feature_scale_levels: dict) -> list[tuple[str, object]]:
    transformers = []
    if feature_scale_levels["numerical"]:
        transformers.append(
            ("numerical", numerical_preprocessing(), feature_scale_levels["numerical"])
        )
    if feature_scale_levels["ordinal"]:
        transformers.append(
            ("ordinal", ordinal_preprocessing(), feature_scale_levels["ordinal"])
        )
    if feature_scale_levels["categorical"]:
        transformers.append(
            (
                "categorical",
                categorical_preprocessing(),
                feature_scale_levels["categorical"],
            )
        )
    return [("preprocessor", ColumnTransformer(transformers))]


def get_sampling_step(method) -> list[tuple[str, object]]:
    """
    Get a sampling method as a pipeline step.

    Sampling methods are applied to address severe class imbalance in datasets by oversampling the minority class.

    Args:
        method: String representation of the sampler.

    Returns:
        list[tuple[str, object]]: A pipeline-step-like object to be integrated into the model pipeline.

    Raises:
        NotImplementedError: If the specified sampling method is not implemented.

    Example:
        >>> get_sampling_step('smote')
        [('sampling', SMOTE())]

    Note:
        Available sampling methods can be found in the 'sample_methods' dictionary.
    """

    if method is None:
        return []
    elif method not in sample_methods.keys():
        raise NotImplementedError(f"The sampling method '{method}' is not implemented.")
    else:
        return [("sampling", sample_methods[method])]


def get_feature_selection_step(method):
    """
    Get a feature selection method as a pipeline step.

    Feature selection methods are applied to prevent overfitting in datasets with a large number of features
    and a low sample size.

    Args:
        method: String representation of the feature selector.

    Returns:
        list[tuple[str, object]]: A pipeline-step-like object to be integrated into the model pipeline.

    Raises:
        NotImplementedError: If the specified feature selection method is not implemented.

    Example:
        >>> get_feature_selection_step('lasso')
        [('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', C=0.8, solver='liblinear')))]

    Note:
        Available feature selection methods can be found in the 'feature_selection_methods' dictionary.
    """
    if method is None:
        return []
    elif method not in feature_selection_methods.keys():
        raise NotImplementedError(
            f"The feature selection method '{method}' is not implemented."
        )
    else:
        return [("feature_selection", feature_selection_methods[method])]


def numerical_preprocessing():
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("imputer", SimpleImputer(strategy="mean"))]
    )
    return pipe


def ordinal_preprocessing():
    pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    return pipe


def categorical_preprocessing():
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    sparse_output=False, drop="if_binary", handle_unknown="ignore"
                ),
            ),
        ]
    )
    return pipe


def build_model(
    feature_scale_levels: dict,
    sample_method: str,
    feature_selection_method: str,
    estimator_name: str,
):
    """
    Build a classification model pipeline based on specified components.

    Args:
        feature_scale_levels (dict): Mapping of features to their corresponding scale level.
        sample_method (str): String representation of the sampling method.
        feature_selection_method (str): String representation of the feature selection method.
        estimator_name (str): String representation of the estimator.

    Returns:
        Pipeline: A machine learning model pipeline.

    Raises:
        NotImplementedError: If the specified estimator is not implemented.

    Example:
        >>> build_model({'numerical': ['feature1'], 'ordinal': [], 'categorical': ['feature2']},
        ...             'smote', 'lasso', 'random_forest')
        Pipeline(steps=[('preprocessor', ...), ('sampling', SMOTE()), ('feature_selection', ...), ('estimator', RandomForestClassifier(n_jobs=-1))])

    Note:
        This function assembles a classification model pipeline by combining preprocessing steps,
        sampling, feature selection, and the chosen estimator.
    """
    preprocessor = get_preprocessing_steps(feature_scale_levels)
    sampler = get_sampling_step(sample_method)
    feature_selector = get_feature_selection_step(feature_selection_method)
    estimator = get_estimator(estimator_name)
    return Pipeline(preprocessor + sampler + feature_selector + estimator)
