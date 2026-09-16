from pathlib import Path

import joblib
import pandas as pd

from train import DATA_PATH, FEATURES, MODEL_PATH, train_model


class MachinePredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists() or self._needs_retraining(model_path):
            train_model()
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.feature_medians = bundle.get("feature_medians", {})

    @staticmethod
    def _needs_retraining(model_path: Path) -> bool:
        bundle = joblib.load(model_path)
        signature = bundle.get("data_signature")
        current_signature = (DATA_PATH.stat().st_size, DATA_PATH.stat().st_mtime_ns)
        return signature != current_signature

    def predict(self, sensor_values: dict[str, float]) -> dict:
        readings = {feature: float(sensor_values[feature]) for feature in self.features}
        frame = pd.DataFrame([readings], columns=self.features)
        probability = float(self.model.predict_proba(frame)[0][1])
        global_importance = self.model.named_steps["classifier"].feature_importances_
        impact = []
        for feature, importance in zip(self.features, global_importance):
            baseline = readings.copy()
            baseline[feature] = self.feature_medians.get(feature, readings[feature])
            baseline_probability = float(self.model.predict_proba(pd.DataFrame([baseline], columns=self.features))[0][1])
            impact.append(
                {
                    "feature": feature,
                    "importance": float(importance),
                    "probability_change": probability - baseline_probability,
                    "value": readings[feature],
                }
            )
        impact.sort(key=lambda item: abs(item["probability_change"]), reverse=True)
        if probability >= 0.7:
            status = "HIGH RISK"
        elif probability >= 0.4:
            status = "MEDIUM RISK"
        else:
            status = "LOW RISK"
        return {
            "readings": readings,
            "failure_probability": probability,
            "status": status,
            "feature_impact": impact,
        }
