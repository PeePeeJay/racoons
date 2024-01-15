from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_float_dtype,
    is_categorical_dtype,
    is_integer_dtype,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

######################## OPEN ISSUES ##################################################
# TODO: Documentation!
# TODO: Testing!
# TODO: Add feature importance to reporting table
# TODO: refit on f1 in cross-validation
# TODO: class balancing strategies
#######################################################################################

sklearn.set_config(transform_output="pandas")

classifiers = {
    "logistic_regression": LogisticRegression(n_jobs=-1),
    "random_forest": RandomForestClassifier(n_jobs=-1),
    "ada_boost": AdaBoostClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "k_neighbors": KNeighborsClassifier(n_jobs=-1),
}

sample_methods = {
    "smote": SMOTE(),
    "adasyn": ADASYN(),
    "random_oversampling": RandomOverSampler(),
}

feature_selection_methods = {
    "lasso": SelectFromModel(
        LogisticRegression(penalty="l1", C=0.8, solver="liblinear")
    ),
}

supported_scale_levels = [
    "numerical",
    "ordinal",
    "categorical",
]


def features_and_targets_from_dataframe(df, feature_cols, target_cols):
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


def get_estimator(estimator_name: str):
    if estimator_name not in classifiers.keys():
        raise NotImplementedError(
            f"The classifier '{estimator_name}' is not implemented."
        )
    else:
        return [("estimator", classifiers[estimator_name])]


def get_preprocessing_steps(feature_scale_levels):
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


def get_sampling_step(method):
    if method is None:
        return []
    elif method not in sample_methods.keys():
        raise NotImplementedError(f"The sampling method '{method}' is not implemented.")
    else:
        return [("sampling", sample_methods[method])]


def get_feature_selection_steps(method):
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
            (
                "encoder",
                OneHotEncoder(
                    sparse_output=False, drop="if_binary", handle_unknown="ignore"
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    return pipe


def get_scale_level(feature: pd.Series):
    if isinstance(feature.dtype, float):
        return "numerical"
    elif isinstance(feature.dtype, pd.Int64Dtype):
        return "ordinal"
    elif isinstance(feature.dtype, pd.CategoricalDtype):
        return "categorical"
    else:
        print(f"The feature {feature} has an unsupported dtype {feature.dtype}.")


def build_model(
    feature_scale_levels: dict,
    sample_method: str,
    feature_selection_method: str,
    estimator_name: str,
):
    preprocessor = get_preprocessing_steps(feature_scale_levels)
    sampler = get_sampling_step(sample_method)
    feature_selector = get_feature_selection_steps(feature_selection_method)
    estimator = get_estimator(estimator_name)
    return Pipeline(preprocessor + sampler + feature_selector + estimator)


def hyper_parameter_optimization(model, X, y):
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


def get_param_grid(model):
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
    elif classifier == "k_neighbors":
        param_grid["estimator__n_neighbors"] = [2, 5, 10, 50]
        param_grid["estimator__n_jobs"] = [-1]
    return param_grid


def cross_validate_model(model, X, y):
    tprs = []
    aucs = []
    f1 = []
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
    return tprs, aucs, f1


def metrics_from_cv_result(cv_result: tuple):
    tprs, aucs, f1 = cv_result
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


def make_report_df(sampling, feature_selection, grid_search=False):
    columns = [
        "Outcome",
        "Class distribution",
        "Covariates",
        "Model",
        "AUC_mean",
        "AUC_std",
        "f1_mean",
        "f1_std",
        "plot_path",
    ]

    if feature_selection is not None:
        columns.insert(columns.index("Model"), "Number of selected covariates")
        columns.insert(
            columns.index("Number of selected covariates") + 1,
            "Feature selection method",
        )

    if sampling is not None:
        columns.insert(columns.index("Model"), "Sampling method")

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
    figure_path,
    sampling=None,
    feature_selection=None,
    selected_features=None,
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
        figure_path,
    ]
    if feature_selection is not None:
        row.insert(row.index(estimator_name), feature_selection)
        row.insert(row.index(feature_selection) + 1, len(selected_features))
    if sampling is not None:
        row.insert(row.index(estimator_name), sampling)
    return row


def plot_roc_curve_from_cv_metrics(cv_result_metrics, plot_title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        cv_result_metrics["mean_fpr"],
        cv_result_metrics["mean_tpr"],
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)"
        % (cv_result_metrics["mean_auc"], cv_result_metrics["std_auc"]),
        lw=2,
        alpha=0.8,
    )

    tprs_upper = np.minimum(
        cv_result_metrics["mean_tpr"] + cv_result_metrics["std_tpr"], 1
    )
    tprs_lower = np.maximum(
        cv_result_metrics["mean_tpr"] - cv_result_metrics["std_tpr"], 0
    )
    ax.fill_between(
        cv_result_metrics["mean_fpr"],
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{plot_title}",
    )
    ax.title.set_size(8)
    ax.axis("square")
    ax.legend(loc="lower right")
    return fig


def classification(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    feature_selection_method: str,
    sample_method: str,
    estimators: list[str],
    output_path: Path,
    **kwargs,
):
    # setup output folder
    output_path.mkdir(exist_ok=True)
    output_folder = (
        output_path
        / f"multivariate_analysis_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    )
    output_folder.mkdir(exist_ok=True)

    # get features and targets
    features, targets, feature_scale_levels = features_and_targets_from_dataframe(
        df, feature_cols, target_cols
    )

    # iterate over targets and features
    report_df = make_report_df(sample_method, feature_selection_method)
    with tqdm(total=(targets.shape[1]) * len(classifiers)) as pbar:
        plot_index = 0
        for target in targets:
            for estimator in estimators:
                estimator_name = type(get_estimator(estimator)[0][1]).__name__
                model = build_model(
                    feature_scale_levels,
                    sample_method,
                    feature_selection_method,
                    estimator,
                )
                cv_result = cross_validate_model(model, df[feature_cols], df[target])
                cv_result_metrics = metrics_from_cv_result(cv_result)
                # save roc-auc plot
                roc_curve_plot = plot_roc_curve_from_cv_metrics(
                    cv_result_metrics,
                    plot_title=f"Classification of {target} using " f"{estimator_name}",
                )
                figure_path = output_folder / (f"roc_auc_{plot_index}.png")
                roc_curve_plot.savefig(figure_path, dpi=300)
                plt.close(roc_curve_plot)

                # save report
                selected_features = model["estimator"].feature_names_in_
                negative_samples = (~df[target]).sum()
                positive_samples = (df[target]).sum()
                mean_auc = cv_result_metrics["mean_auc"]
                std_auc = cv_result_metrics["std_auc"]
                mean_f1 = cv_result_metrics["mean_f1"]
                std_f1 = cv_result_metrics["std_f1"]

                report_df.loc[len(report_df.index)] = update_report(
                    target=target,
                    features=feature_cols,
                    negative_samples=negative_samples,
                    positive_samples=positive_samples,
                    estimator_name=estimator_name,
                    mean_auc=mean_auc,
                    std_auc=std_auc,
                    mean_f1=mean_f1,
                    std_f1=std_f1,
                    figure_path=figure_path,
                    sampling=sample_method,
                    feature_selection=feature_selection_method,
                    selected_features=selected_features,
                )
                pbar.update(1)
                plot_index += 1
            report_df.to_excel(output_folder / "report.xlsx")
            report_df.to_csv(output_folder / "report.csv", sep=";")

        report_df.to_excel(output_folder / "report.xlsx")
        report_df.to_csv(output_folder / "report.csv", sep=";")
        return report_df


def grid_search_classification(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    feature_selection_method: str,
    sample_method: str,
    estimators: list[str],
    output_path: Path,
    **kwargs,
):
    # setup output folder
    output_path.mkdir(exist_ok=True)
    output_folder = (
        output_path
        / f"multivariate_analysis_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    )
    output_folder.mkdir(exist_ok=True)

    # get features and targets
    features, targets, feature_scale_levels = features_and_targets_from_dataframe(
        df, feature_cols, target_cols
    )

    # iterate over targets and features
    report_df = make_report_df(
        sample_method, feature_selection_method, grid_search=True
    )
    with tqdm(total=(targets.shape[1])) as pbar:
        # find best performing model by cross validation
        plot_index = 0
        for target in targets:
            cv_df = pd.DataFrame(columns=["Model", "AUC"])
            print("Initial cross-validation...")
            for estimator in estimators:
                # remove feature selection steps from model for initial cv
                model = build_model(
                    feature_scale_levels, sample_method, None, estimator
                )
                initial_cv_result = cross_validate_model(
                    model, df[feature_cols], df[target]
                )

                cv_result_metrics = metrics_from_cv_result(initial_cv_result)
                cv_df.loc[len(report_df.index)] = [
                    estimator,
                    cv_result_metrics["mean_auc"],
                ]
            print("Done.")
            best_cv_estimator = cv_df.sort_values(by="AUC", ascending=False)["Model"][0]
            grid_search_model = build_model(
                feature_scale_levels,
                sample_method,
                feature_selection_method,
                best_cv_estimator,
            )
            print("Hyper-parameter optimization...")
            best_params, mean_score, std_score = hyper_parameter_optimization(
                grid_search_model,
                df[feature_cols],
                df[target],
            )
            print("done.")
            # run cross valiadtion with best model
            optimized_model = build_model(
                feature_scale_levels,
                sample_method,
                feature_selection_method,
                best_cv_estimator,
            )
            optimized_model.set_params(**best_params)
            cv_result = cross_validate_model(
                optimized_model, df[feature_cols], df[target]
            )
            cv_result_metrics = metrics_from_cv_result(cv_result)

            # save roc-auc plot
            roc_curve_plot = plot_roc_curve_from_cv_metrics(
                cv_result_metrics,
                plot_title=f"Classification of {target} using " f"{best_cv_estimator}",
            )
            figure_path = output_folder / (f"roc_auc_{plot_index}.png")
            roc_curve_plot.savefig(figure_path, dpi=300)
            plt.close(roc_curve_plot)

            # save report
            selected_features = model["estimator"].feature_names_in_
            negative_samples = (~df[target]).sum()
            positive_samples = (df[target]).sum()
            mean_auc = cv_result_metrics["mean_auc"]
            std_auc = cv_result_metrics["std_auc"]

            report_df.loc[len(report_df.index)] = update_report(
                target=target,
                features=feature_cols,
                negative_samples=negative_samples,
                positive_samples=positive_samples,
                estimator_name=best_cv_estimator,
                mean_auc=mean_auc,
                std_auc=std_auc,
                mean_f1=mean_score,
                std_f1=std_score,
                figure_path=figure_path,
                sampling=sample_method,
                feature_selection=feature_selection_method,
                selected_features=selected_features,
            )
            pbar.update(1)
            plot_index += 1
            report_df.to_excel(output_folder / "report.xlsx")
            report_df.to_csv(output_folder / "report.csv", sep=";")
        report_df.to_excel(output_folder / "report.xlsx")
        report_df.to_csv(output_folder / "report.csv", sep=";")
        return report_df
