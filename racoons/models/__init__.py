from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

classifiers = {
    "logistic_regression": LogisticRegression(n_jobs=-1),
    "random_forest": RandomForestClassifier(n_jobs=-1),
    "ada_boost": AdaBoostClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "xgboost": XGBClassifier()
    # "k_neighbors": KNeighborsClassifier(n_jobs=-1),
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
