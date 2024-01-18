import pytest
from imblearn.pipeline import Pipeline

from racoons.data_utils import features_and_targets_from_dataframe
from racoons.models import classifiers, sample_methods, feature_selection_methods
from racoons.models.model_builder import get_estimator, get_preprocessing_steps, get_sampling_step, \
    get_feature_selection_step, build_model


def test_get_estimator():
    for classifier in classifiers.keys():
        estimator = get_estimator(classifier)
        assert isinstance(estimator[0][1], type(classifiers[classifier]))
    with pytest.raises(NotImplementedError):
        get_estimator("unknown_estimator")


def test_get_preprocessing_steps(classification_data):
    df, target_cols, feature_cols = classification_data
    X, y, scale_levels = features_and_targets_from_dataframe(df, feature_cols, target_cols)
    preprocessing_steps = get_preprocessing_steps(scale_levels)
    transformers = preprocessing_steps[0][1].transformers
    assert transformers[0][0] == "numerical"
    assert transformers[0][2] == [f"feature_{i}" for i in range(2, len(feature_cols))]
    assert transformers[1][0] == "ordinal"
    assert transformers[1][2] == ["feature_0"]
    assert transformers[2][0] == "categorical"
    assert transformers[2][2] == ["feature_1"]


def test_get_sampling_steps():
    for sample_method in sample_methods.keys():
        sampler = get_sampling_step(sample_method)
        assert isinstance(sampler[0][1], type(sample_methods[sample_method]))
    with pytest.raises(NotImplementedError):
        get_sampling_step("unknown_sample_method")


def test_get_feature_selection_steps():
    for feature_selection_method in feature_selection_methods.keys():
        selector = get_feature_selection_step(feature_selection_method)
        assert isinstance(selector[0][1], type(feature_selection_methods[feature_selection_method]))
    with pytest.raises(NotImplementedError):
        get_feature_selection_step("unknown_sample_method")


def test_build_model(classification_data):
    df, target_cols, feature_cols = classification_data
    X, y, scale_levels = features_and_targets_from_dataframe(df, feature_cols, target_cols)

    sample_method = "smote"
    feature_selection_method = "lasso"
    estimator_name = "random_forest"

    model = build_model(scale_levels, sample_method, feature_selection_method, estimator_name)

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "sampling" in model.named_steps
    assert "feature_selection" in model.named_steps
    assert "estimator" in model.named_steps
    assert model["sampling"].__class__.__name__ == "SMOTE"
    assert model["feature_selection"].__class__.__name__ == "SelectFromModel"
    assert model["estimator"].__class__.__name__ == "RandomForestClassifier"
    assert model["estimator"].n_jobs == -1
