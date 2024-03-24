from racoons.modelling.models import Model
from racoons.config import configurable
from racoons.modelling import estimators
from racoons.modelling import tasks, target_types


class EstimatorInterface:
    def fit(self):
        pass

    def predict(self):
        pass

    def transform(self):
        pass


class PipelineEstimator(EstimatorInterface):
    def __init__(self, model: Model):
        self._pipeline = self._assemble_pipeline()

    def _assemble_pipeline(self):
        return None

    def fit(self):
        return

def get_pipeline_steps():
    pass

