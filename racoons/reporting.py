import pandas as pd


def make_report_df(sampling: str, feature_selection: str, grid_search=False):
    """
    Creates an empty dataframe to store the results of the classification task.
    Args:
        sampling (str): sample method string representation
        feature_selection (str): feature selection string representation
        grid_search (bool): grid search optimization was conducted

    Returns:
        pd.DataFrame: empty dataframe to store the classification results
    """
    columns = [
        "Outcome",
        "Class distribution",
        "Covariates",
        "Model",
        "AUC_mean",
        "AUC_std",
        "f1_mean",
        "f1_std",
        "feature_importance",
        "roc_curve_plot_path",
        "feature_importance_plot_path",
    ]

    if feature_selection is not None:
        columns.insert(columns.index("Model"), "Number of selected covariates")
        columns.insert(
            columns.index("Number of selected covariates") + 1,
            "Feature selection method",
        )

    if sampling is not None:
        columns.insert(columns.index("Model"), "Sampling method")

    if grid_search:
        columns.insert(columns.index("Model"), "Best_params")
    return pd.DataFrame(columns=columns)


def update_report(
    target,
    features,
    negative_samples,
    positive_samples,
    estimator_name,
    mean_auc,
    std_auc,
    mean_f1,
    std_f1,
    feature_importance_csv,
    roc_plot_path,
    feature_importance_plot_path,
    sampling=None,
    feature_selection=None,
    selected_features=None,
    best_params=None,
):
    row = [
        target,
        f"positive: {positive_samples}, negative: {negative_samples}",
        features,
        estimator_name,
        mean_auc,
        std_auc,
        mean_f1,
        std_f1,
        feature_importance_csv,
        roc_plot_path,
        feature_importance_plot_path,
    ]
    if feature_selection is not None:
        row.insert(row.index(estimator_name), feature_selection)
        row.insert(row.index(feature_selection), len(selected_features))
    if sampling is not None:
        row.insert(row.index(estimator_name), sampling)
    if best_params:
        row.insert(row.index(estimator_name), f"{best_params}")
    return row
