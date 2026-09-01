import joblib
from sklearn.ensemble import RandomForestRegressor
from .base_predictor import BasePredictor

class RandomForestPredictor(BasePredictor):
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = self._build_model()

    def _build_model(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        self.model.fit(X_train, y_train.ravel())

        # Calculate validation score
        val_score = self.model.score(X_val, y_val.ravel())

        return {'val_r2': val_score}

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)