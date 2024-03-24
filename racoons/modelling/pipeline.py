from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import inspect


class Step:
    def __init__(self, name="step_name"):
        self.name = name

    def get_step(self):
        return [(self.name, steps[self.name])]


def xgboost_classifier(n_estimators: int = int(500)):
    return XGBClassifier(n_estimators)


def lasso_feature_selection(
        C: float = 0.8,
):
    estimators = {
        LogisticRegression(penalty="l1")
    }
    return


steps = {
    "predictor": {
        "classifier": {
            "xgboost classifier": xgboost_classifier()
        },
        "feature selection": {
            "lasso": {
                "logistic regression": lasso_feature_selection()
            }
        }
    }
}
