from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, f1_score, auc
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

from racoons.models import classifiers, feature_selection_methods


def hyper_parameter_optimization(
    model: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[dict[Any, Any, Any], float, float]:
    """
    Optimize the hyperparameters of the given model using grid search cross-validation.

    Args:
        model (Pipeline): The machine learning model to be optimized.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.

    Returns:
        Tuple[Dict[Any, Any, Any], float, float]: A tuple containing:
            - The best hyperparameters found during optimization.
            - Mean F1 score on the test sets for the best hyperparameters.
            - Standard deviation of F1 score on the test sets for the best hyperparameters.

    Raises:
        Any specific exceptions or errors.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> best_params, mean_f1, std_f1 = hyper_parameter_optimization(
        ...     RandomForestClassifier(),
        ...     X_train,
        ...     y_train
        ... )

    Note:
        - This function performs hyperparameter optimization using grid search with cross-validation.
        - The search space for hyperparameters is defined by the get_param_grid() function.
        - Cross-validation is conducted using RepeatedStratifiedKFold with 5 splits and 2 repeats.
        - Scoring is based on both F1 score and accuracy, with F1 score used for refitting.
        - The function returns the best hyperparameters and associated mean and standard deviation of the F1 score.

    """
    params = get_param_grid(model)
    grid_cv = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2),
        scoring=["f1", "accuracy"],
        refit="f1",
        n_jobs=-1,
    )
    grid_cv.fit(X, y)
    return (
        grid_cv.best_params_,
        grid_cv.cv_results_["mean_test_f1"][grid_cv.best_index_],
        grid_cv.cv_results_["std_test_f1"][grid_cv.best_index_],
    )


def get_param_grid(model: Pipeline):
    """
    Get the hyperparameter grid for the specified model.

    The hyperparameter grid contains candidate values for hyperparameters to be used
    in hyperparameter optimization during model training.

    Args:
        model (Pipeline): The machine learning model pipeline.

    Returns:
        dict: A dictionary representing the hyperparameter grid.

    Example:
        >> from imblearn.pipeline import Pipeline
        >> from sklearn.ensemble import RandomForestClassifier
        >> from sklearn.feature_selection import SelectFromModel
        >> model = Pipeline(steps=[
        ...     ('preprocessor', ...),
        ...     ('sampling', ...),
        ...     ('feature_selection', SelectFromModel(RandomForestClassifier())),
        ...     ('estimator', RandomForestClassifier())
        ... ])
        >>> get_param_grid(model)
        {'feature_selection__estimator__C': [0.01, 0.1, 1, 10, 100],
         'feature_selection__estimator__solver': ['liblinear'],
         'estimator__n_estimators': [100, 500, 1000],
         'estimator__n_jobs': [-1]}

    Note:
        This function generates a hyperparameter grid based on the type of classifier in the pipeline.
        It is used for hyperparameter optimization using GridSearchCV.
    """
    param_grid = {}
    classifier = list(classifiers.keys())[
        list(classifiers.values()).index(model["estimator"])
    ]
    if "feature_selection" in model.named_steps.keys():
        feature_selection_method = list(feature_selection_methods.keys())[
            list(feature_selection_methods.values()).index(model["feature_selection"])
        ]
        if feature_selection_method == "lasso":
            param_grid["feature_selection__estimator__C"] = [0.01, 0.1, 1, 10, 100]
            param_grid["feature_selection__estimator__solver"] = ["liblinear"]
            # param_grid["feature_selection__estimator__n_jobs"] = [-1]
    if classifier == "logistic_regression":
        param_grid["estimator__penalty"] = ["l1", "l2"]
        param_grid["estimator__solver"] = ["liblinear"]
        param_grid["estimator__C"] = [0.3, 0.5, 0.8, 1, 10, 100]
        # param_grid["estimator__n_jobs"] = [-1]
    elif classifier == "random_forest":
        param_grid["estimator__n_estimators"] = [100, 500, 1000]
        param_grid["estimator__n_jobs"] = [-1]
    elif classifier == "ada_boost":
        param_grid["estimator__learning_rate"] = np.arange(0.1, 2.1, 0.1)
        param_grid["estimator__n_estimators"] = [10, 50, 100, 500, 1000]
    elif classifier == "gradient_boosting":
        param_grid["estimator__learning_rate"] = np.arange(0.1, 2.1, 0.1)
        param_grid["estimator__n_estimators"] = [10, 50, 100, 500, 1000]
    elif classifier == "decision_tree":
        param_grid["estimator__criterion"] = ["gini", "log_loss", "entropy"]
    elif classifier == "xgboost":
        param_grid["estimator__n_estimators"] = [10, 50, 100, 500, 1000]
    # elif classifier == "k_neighbors":
    #     param_grid["estimator__n_neighbors"] = [2, 5, 10, 50]
    #     param_grid["estimator__n_jobs"] = [-1]
    return param_grid


def cross_validate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Perform cross-validation for a given classification model.

    Args:
        model (Pipeline): The machine learning model pipeline.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.

    Returns:
        tuple: A tuple containing lists of true positive rates, area under the ROC curve, F1 score and a dataframe of feature importances for each fold.

    Raises:
        None

    Example:
        >> from imblearn.pipeline import Pipeline
        >> from sklearn.ensemble import RandomForestClassifier
        >> from sklearn.model_selection import train_test_split
        >> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        >> model = Pipeline(steps=[
        ...     ('preprocessor', ...),
        ...     ('sampling', ...),
        ...     ('feature_selection', SelectFromModel(RandomForestClassifier())),
        ...     ('estimator', RandomForestClassifier())
        ... ])
        >> tprs, aucs, f1_scores = cross_validate_model(model, X_train, y_train)

    Note:
        This function uses StratifiedKFold with 10 folds for cross-validation, and it returns lists of true positive rates,
        area under the ROC curve, F1 scores and feature importances for each fold.
    """
    tprs = []
    aucs = []
    f1 = []
    feature_importance = pd.DataFrame()
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(10)
    for fold, (train, test) in enumerate(cv.split(X, y)):
        X_train = X.loc[train, :]
        X_test = X.loc[test, :]
        y_train = y[train]
        y_test = y[test]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        f1.append(f1_score(y_test, y_pred))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
        feature_importance = pd.concat(
            [feature_importance, get_feature_importance(model)], axis=0
        )

    return tprs, aucs, f1, feature_importance


def metrics_from_cv_result(cv_result: tuple):
    """
    Computes results from the cv
    Args:
        cv_result (tuple): true positive rates, aucs, f1-scores

    Returns:
        dict: means and standard of the cross-validation metrics
    """
    tprs, aucs, f1, _ = cv_result
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    mean_f1 = np.mean(f1)
    std_f1 = np.std(f1)
    metrics = {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "std_tpr": std_tpr,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
    }
    return metrics


def get_feature_importance(model):
    if hasattr(model["estimator"], "coef_"):
        return pd.DataFrame(
            model["estimator"].coef_,
            columns=model["estimator"].feature_names_in_,
        )
    elif hasattr(model["estimator"], "feature_importances_"):
        return pd.DataFrame(
            [model["estimator"].feature_importances_],
            columns=model["estimator"].feature_names_in_,
        )
    else:
        print(
            f"The model estimator {model['estimator']} does not support feature importance inference."
        )
        return None
