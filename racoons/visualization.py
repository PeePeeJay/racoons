import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_feature_importances(feature_importances: pd.DataFrame, title=None) -> plt.Figure:
    df = feature_importances.reindex(
        feature_importances.mean().abs().sort_values(ascending=False).index, axis=1
    )
    if title is None:
        title = "Feature importance"
    if df.shape[1] > 20:
        data = df.loc[:, df.columns[:20]]
    else:
        data = df
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.barplot(data=data, orient="h", color="b", saturation=1, ax=ax)
    ax.axvline(x=0, color=".5")
    ax.xaxis.set_label("Feature importance")
    ax.set_title(f"{title}")
    ax.tick_params(axis="y", labelsize=5)
    return fig


def plot_roc_curve_from_cv_metrics(cv_result_metrics: dict, plot_title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        cv_result_metrics["mean_fpr"],
        cv_result_metrics["mean_tpr"],
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)"
        % (cv_result_metrics["mean_auc_preds"], cv_result_metrics["std_auc_preds"]),
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
