from racoons.config import configurable


class Model:
    def __init__(self, data, estimator):
        self.data = data
        self.estimator = None

    def _build_estimator(self, estimator_class=None):
        self.estimator = estimator_class(self)

    # def _validate(self, validation=None):
    #     self.scores, self.params = validation(self)






