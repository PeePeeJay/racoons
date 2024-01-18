import numpy as np

from racoons.models import classifiers
from racoons.models.validation import get_param_grid, cross_validate_model, get_feature_importance
from racoons.models.model_builder import build_model
from racoons.models.classification import multivariate_classification, grid_search_multivariate_classification, univariate_classification
from racoons.visualization import plot_feature_importances
from racoons.data_utils import features_and_targets_from_dataframe


def test_feature_importance(classification_data):
    df, target_cols, feature_cols = classification_data
    X, y, scale_levels = features_and_targets_from_dataframe(df, feature_cols, target_cols)

    sample_method = "smote"
    feature_selection_method = "lasso"
    estimator_name = "logistic_regression"

    model = build_model(scale_levels, sample_method, feature_selection_method, estimator_name)
    model.fit(X, y)
    feature_importance = get_feature_importance(model)
    plot_feature_importances(feature_importance)
    assert feature_importance.columns.tolist() == model["estimator"].feature_names_in_.tolist()
    assert not feature_importance.empty


class TestClassification:
    def test_cross_validate_model(self, classification_data):
        df, target_cols, feature_cols = classification_data
        X, y, scale_levels = features_and_targets_from_dataframe(df, feature_cols, target_cols)

        sample_method = "smote"
        feature_selection_method = "lasso"
        estimator_name = "random_forest"

        model = build_model(scale_levels, sample_method, feature_selection_method, estimator_name)

        # Test the cross_validate_model function
        tprs, aucs, f1_scores, feature_importances = cross_validate_model(model, X, y["outcome"])


        # Check if the output lists have the correct lengths (10 folds)
        assert len(tprs) == len(aucs) == len(f1_scores) == 10

        # Check if feature importances is not empty
        assert not feature_importances.empty

        # Check if each element in the lists is a NumPy array
        assert all(isinstance(tpr, np.ndarray) for tpr in tprs)
        assert all(isinstance(auc_, np.float64) for auc_ in aucs)
        assert all(isinstance(f1_score_, np.float64) for f1_score_ in f1_scores)

    def test_multivariate_classification(self, classification_data, output_path, tmp_path):
        out_path = tmp_path
        df, target_cols, feature_cols = classification_data

        result_df = multivariate_classification(
            df=df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            feature_selection_method="lasso",
            sample_method="smote",
            estimators=classifiers.keys(),
            output_path=out_path,
        )

        # Validate the result
        assert len(result_df) == len(target_cols) * len(classifiers)
        assert not result_df.empty

    def test_univariate_classification(self, classification_data, tmp_path, output_path):
        out_path = tmp_path
        df, target_cols, feature_cols = classification_data

        result_df = univariate_classification(
            df=df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            sample_method="smote",
            estimators=classifiers.keys(),
            output_path=out_path,
        )

        # Validate the result
        assert len(result_df) == len(target_cols) * len(classifiers) * len(feature_cols)
        assert not result_df.empty


class TestGridSearchClassification:
    def test_get_param_grid(self, classification_data):
        for classifier_name, classifier in classifiers.items():
            df, target_cols, feature_cols = classification_data
            X, y, scale_levels = features_and_targets_from_dataframe(df, feature_cols, target_cols)
            model = build_model(
                feature_scale_levels=scale_levels,
                feature_selection_method=None,
                sample_method="smote",
                estimator_name=classifier_name
            )
            param_grid = get_param_grid(model)

            if classifier_name == "logistic_regression":
                assert "estimator__penalty" in param_grid
                assert "estimator__solver" in param_grid
                assert "estimator__C" in param_grid
            elif classifier_name == "random_forest":
                assert "estimator__n_estimators" in param_grid
                assert "estimator__n_jobs" in param_grid
            elif classifier_name == "ada_boost":
                assert "estimator__learning_rate" in param_grid
                assert "estimator__n_estimators" in param_grid
            elif classifier_name == "gradient_boosting":
                assert "estimator__learning_rate" in param_grid
                assert "estimator__n_estimators" in param_grid
            elif classifier_name == "decision_tree":
                assert "estimator__criterion" in param_grid
            elif classifier_name == "xgboost":
                assert "estimator__n_estimators" in param_grid
            # elif classifier_name == "k_neighbors":
            #     assert "estimator__n_neighbors" in param_grid
            #     assert "estimator__n_jobs" in param_grid

            model = build_model(
                feature_scale_levels=scale_levels,
                feature_selection_method="lasso",
                sample_method="smote",
                estimator_name=classifier_name
            )
            param_grid = get_param_grid(model)
            assert "feature_selection__estimator__C" in param_grid
            assert "feature_selection__estimator__solver" in param_grid

    def test_grid_search_classification(self, classification_data, tmp_path, output_path):
        df, target_cols, feature_cols = classification_data
        out_path = tmp_path
        result_df = grid_search_multivariate_classification(
            df=df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            feature_selection_method="lasso",
            sample_method="smote",
            estimators=classifiers.keys(),
            output_path=out_path,
        )

        # Validate the result
        assert len(result_df) == len(target_cols)
        assert not result_df.empty


