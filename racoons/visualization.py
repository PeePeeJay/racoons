import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_feature_importances(feature_importances: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(9, 7))
    sns.boxplot(data=feature_importances, orient="h", color="cyan", saturation=0.5)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Feature importance")
    plt.title("Feature importance and its variability")
    plt.subplots_adjust(left=0.3)
    return fig


def plot_roc_curve_from_cv_metrics(cv_result_metrics: dict, plot_title: str):
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
