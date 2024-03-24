from racoons.modelling.models import Model
from racoons.modelling.estimators import EstimatorInterface


class TestModel:
    def test_model__init__(self, config_model, dataset_mock):
        model = Model(dataset_mock)
        assert isinstance(model.data, type(dataset_mock))


class TestEstimator:
    def test_estimator_interface(self):
        method_list = [method for method in dir(EstimatorInterface) if method.startswith('_') is False]
        assert all(x in method_list for x in ["fit", "transform", "predict"])

    def test__get_step(self, test_step):
        pass


class TestValidation:
    pass



