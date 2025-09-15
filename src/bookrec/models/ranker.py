import os
import joblib


class BookRanker:

    def __init__(self, model_path: str):
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None

    def predict_proba(self, features: list[list[float]]) -> list[float]:
        if self.model:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(features)[:, 1]
            else:
                return self.model.predict(features)
        return [0.5] * len(features)

    def predict(self, features: list[list[float]]) -> list[float]:
        if self.model:
            return self.model.predict(features)
        return [0] * len(features)