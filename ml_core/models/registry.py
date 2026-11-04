"""Реестр моделей машинного обучения."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_REGISTRY = {
    "LR": lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    ),
    "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
    "XGB": lambda: XGBClassifier(eval_metric="logloss", random_state=42),
}


def get_model(name: str):
    """Возвращает свежий экземпляр модели по имени."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Модель '{name}' не зарегистрирована. Доступны: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
