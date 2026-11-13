"""Тренер моделей: обучение по реестру, кросс-валидация, сохранение в artifacts/models."""

import json
import os
from datetime import datetime

import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ml_core.core.config import config
from ml_core.models.registry import get_model, MODEL_REGISTRY


class ModelTrainer:
    def __init__(self, models_dir=None, model_names=None):
        self.models_dir = models_dir or config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_names = model_names or list(MODEL_REGISTRY.keys())
        self.models = {name: get_model(name) for name in self.model_names}

    def cross_validate(self, X, y, cv_folds=5, scoring="f1"):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            results[name] = {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores.tolist()}
        return results

    def fit_all(self, X_train, y_train, sample_weight=None):
        """Обучает все модели. Возвращает dict {name: model}."""
        fitted = {}
        for name, model in self.models.items():
            if sample_weight is not None and hasattr(model, "fit"):
                try:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            fitted[name] = model
        return fitted

    def save_model(self, model, model_name, metrics, features):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        joblib.dump(model, model_path)

        metadata = {
            "model_name": model_name,
            "filename": model_filename,
            "timestamp": timestamp,
            "metrics": metrics,
            "features": features,
            "model_type": type(model).__name__,
        }
        with open(os.path.join(self.models_dir, f"{model_name}_{timestamp}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return model_path, metadata
