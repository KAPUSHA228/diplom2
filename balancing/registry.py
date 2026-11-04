"""Реестр методов балансировки: имя → фабрика экземпляра."""

from balancing.base import Balancer
from balancing.smote_variants import (
    SMOTEBalancer,
    BorderlineSMOTEBalancer,
    ADASYNBalancer,
    SMOTETomekBalancer,
    SMOTEENNBalancer,
)
from balancing.class_weight import ClassWeightBalancer


class NoBalancer(Balancer):
    """Базовый вариант: без балансировки."""

    name = "baseline"

    def fit_resample(self, X, y):
        return X, y


BALANCERS = {
    "baseline": NoBalancer,
    "smote": SMOTEBalancer,
    "borderline_smote": BorderlineSMOTEBalancer,
    "adasyn": ADASYNBalancer,
    "smote_tomek": SMOTETomekBalancer,
    "smote_enn": SMOTEENNBalancer,
    "class_weight": ClassWeightBalancer,
}


def get_balancer(name: str) -> Balancer:
    """Возвращает экземпляр балансировщика по имени."""
    if name not in BALANCERS:
        raise KeyError(f"Балансировщик '{name}' не зарегистрирован. Доступны: {list(BALANCERS)}")
    return BALANCERS[name]()
