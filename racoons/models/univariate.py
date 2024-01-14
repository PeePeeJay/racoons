from datetime import datetime

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_float_dtype,
    is_categorical_dtype,
    is_integer_dtype,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
from tqdm import tqdm

######################## OPEN ISSUES ##################################################
# TODO: Documentation!
# TODO: Testing!
# TODO: refit on f1 in cross-validation
# TODO: class balancing strategies
#######################################################################################


classifiers = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "ada_boost": AdaBoostClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "k_neighbors": KNeighborsClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
}

supported_scale_levels = [
    "numerical",
    "ordinal",
    "categorical",
]

sample_methods = {
    "smote": SMOTE(),
    "adasyn": ADASYN(),
    "random_oversampling": RandomOverSampler(),
}


def features_and_targets_from_dataframe(df, feature_cols, target_cols):
    # check scale level of feature columns
    feature_cols_selected = []
    for col in feature_cols:
        if get_scale_level(df[col]) in supported_scale_levels:
            feature_cols_selected.append(col)
        else:
            print(
                f"Feature column {col} has an unsupported dtype {df[col].dtype} and will be dropped.\n"
            )
    print(
        f"{len(feature_cols_selected)} features out of {len(feature_cols)} "
        f"initial features were selected for analysis.\n"
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
    return df.loc[:, feature_cols_selected], df.loc[:, target_cols_selected]


def get_estimator(estimator_name: str):
    if estimator_name not in classifiers.keys():
        raise NotImplementedError(
            f"The classifier '{estimator_name}' is not implemented."
        )
    else:
        return [("estimator", classifiers[estimator_name])]


def get_preprocessing_steps(scale_level):
    if scale_level == "numerical":
        return numerical_preprocessing()
    elif scale_level == "ordinal":
        return ordinal_preprocessing()
    elif scale_level == "categorical":
        return categorical_preprocessing()


def get_sampling_step(method):
    if method is None:
        return []
    elif method not in sample_methods.keys():
        raise NotImplementedError(f"The sampling method '{method}' is not implemented.")
    else:
        return [("sampling", sample_methods[method])]


def numerical_preprocessing():
    steps = [("scaler", StandardScaler()), ("imputer", SimpleImputer(strategy="mean"))]
    return steps


def ordinal_preprocessing():
    steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    return steps


def categorical_preprocessing():
    steps = [
        (
            "encoder",
            OneHotEncoder(
                sparse_output=False, drop="if_binary", handle_unknown="ignore"
            ),
        ),
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]
    return steps


def get_scale_level(feature: pd.Series):
    if is_float_dtype(feature.dtype):
        return "numerical"
    elif is_integer_dtype(feature.dtype):
        return "ordinal"
    elif is_categorical_dtype(feature.dtype):
        return "categorical"
    else:
        print(f"The feature {feature} has an unsupported dtype {feature.dtype}.")


def build_model(estimator_name: str, feature_scale_level: str, sample_method: str):
    pre_processing_steps = get_preprocessing_steps(feature_scale_level)
    sampler = get_sampling_step(sample_method)
    estimator = get_estimator(estimator_name)
    return Pipeline(pre_processing_steps + sampler + estimator)


def cross_validate_model(model, X, y):
    tprs = []
    aucs = []
    f1 = []
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(10)
    for fold, (train, test) in enumerate(cv.split(X, y)):
        X_train = pd.DataFrame(X[train])  # .to_numpy().reshape(-1, 1)
        X_test = pd.DataFrame(X[test])  # .to_numpy().reshape(-1, 1)
        y_train = y[train]
        y_test = y[test]
        # TODO: fit on f1 score because it is more reasonable in the case of unbalanced datasets
        # TODO: report the achieved f1 score in the final report
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


def make_report_df(sampling):
    columns = [
        "Outcome",
        "Class distribution",
        "Covariates",
        "Model",
        "AUC_mean",
        "AUC_std",
        "f1_mean",
        "f2_mean",
        "plot_path",
    ]

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
    if sampling is not None:
        row.insert(row.index(estimator_name), sampling)
    return row


def classification(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    estimators: list[str],
    output_path: Path,
    sample_method=None,
    **kwargs,
):
    # setup output folder
    output_path.mkdir(exist_ok=True)
    output_folder = (
        output_path
        / f"univariate_analysis_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    )
    output_folder.mkdir(exist_ok=True)

    # get features and targets
    features, targets = features_and_targets_from_dataframe(
        df, feature_cols, target_cols
    )

    # iterate over targets and features
    report_df = make_report_df(sample_method)
    with tqdm(total=features.shape[1] * targets.shape[1] * len(estimators)) as pbar:
        plot_index = 0
        for target in targets:
            for feature in features:
                for estimator in estimators:
                    estimator_name = type(get_estimator(estimator)[0][1]).__name__
                    model = build_model(
                        estimator, get_scale_level(df[feature]), sample_method
                    )
                    cv_result = cross_validate_model(model, df[feature], df[target])
                    cv_result_metrics = metrics_from_cv_result(cv_result)

                    # save roc-auc plot
                    roc_curve_plot = plot_roc_curve_from_cv_metrics(
                        cv_result_metrics,
                        plot_title=f"Classification of {target} \n by {feature} \n using "
                        f"{estimator_name}",
                    )
                    figure_path = output_folder / (f"roc_auc_{plot_index}.png")
                    roc_curve_plot.savefig(figure_path, dpi=300)
                    plt.close(roc_curve_plot)

                    # save report
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
                    )
                    pbar.update(1)
                    plot_index += 1
                report_df.to_excel(output_folder / "report.xlsx")
                report_df.to_csv(output_folder / "report.csv", sep=";")
        report_df.to_excel(output_folder / "report.xlsx")
        report_df.to_csv(output_folder / "report.csv", sep=";")
        return report_df
