from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

tasks = ["classification", "regression"]
target_types = ["single", "multi"]

estimators = {
    "classification": {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "ada_boost": AdaBoostClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "decision_tree": DecisionTreeClassifier,
        "xgboost": XGBClassifier
    },
    "regression": {
    }
}
