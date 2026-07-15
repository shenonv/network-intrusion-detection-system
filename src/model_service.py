"""Loads all inference artifacts once and turns raw flows into predictions.

Framework-free on purpose: the websocket server, tests, and batch jobs
all share this one class.
"""
import json

import joblib
import numpy as np

from config import MODELS_DIR
from data_loader import load_feature_names, load_label_encoder, load_scaler

MODEL_FILE = "xgboost.pkl"
DEFAULT_THRESHOLD = 0.5


class ModelService:
    def __init__(self):
        self.model = joblib.load(MODELS_DIR / MODEL_FILE)
        self.scaler = load_scaler()
        self.label_encoder = load_label_encoder()
        self.feature_names = load_feature_names()
        self.classes = list(self.label_encoder.classes_)

        threshold_path = MODELS_DIR / "decision_threshold.json"
        if threshold_path.exists():
            cfg = json.loads(threshold_path.read_text())
            self.threshold = float(cfg["threshold"])
            self.threshold_chosen_by = cfg.get("chosen_by", "unknown")
        else:
            self.threshold = DEFAULT_THRESHOLD
            self.threshold_chosen_by = "default"

    def _to_matrix(self, flows):
        """Validate a list of {feature: value} dicts -> (n, 78) raw matrix."""
        matrix = np.empty((len(flows), len(self.feature_names)), dtype=np.float64)
        for row, flow in enumerate(flows):
            missing = [name for name in self.feature_names if name not in flow]
            if missing:
                raise ValueError(
                    f"Flow {row} is missing {len(missing)} feature(s), "
                    f"e.g. {missing[:5]}"
                )
            values = [flow[name] for name in self.feature_names]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Flow {row} contains NaN or infinite values")
            matrix[row] = values
        return matrix

    def predict(self, flows):
        """flows: list of {feature: raw value} dicts -> list of prediction dicts.

        Works for binary and multiclass models alike: the alert decision is
        benign-vs-any-attack (attack score = 1 - P(Benign) vs the tuned
        threshold), and the label names the most likely attack class.
        """
        X = self.scaler.transform(self._to_matrix(flows))
        proba = self.model.predict_proba(X)
        benign_idx = self.classes.index("Benign") if "Benign" in self.classes else 0

        results = []
        for row in proba:
            attack_probability = float(1.0 - row[benign_idx])
            is_attack = bool(attack_probability >= self.threshold)
            if is_attack:
                attack_scores = row.copy()
                attack_scores[benign_idx] = -1.0
                label = self.classes[int(np.argmax(attack_scores))]
            else:
                label = self.classes[benign_idx]
            results.append({
                "label": label,
                "is_attack": is_attack,
                "attack_probability": round(attack_probability, 4),
                "threshold": self.threshold,
            })
        return results

    def predict_one(self, flow):
        return self.predict([flow])[0]
