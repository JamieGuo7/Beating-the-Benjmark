class BasePredictor:
    """Base class for all prediction models."""

    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError