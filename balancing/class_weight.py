"""Взвешивание классов без генерации синтетики."""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from balancing.base import Balancer


class ClassWeightBalancer(Balancer):
    """
    Не ресемплит данные. Возвращает X, y как есть,
    а веса для обучения доступны через get_sample_weight(y).
    """

    name = "class_weight"

    def fit_resample(self, X, y):
        return X, y

    def get_sample_weight(self, y):
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        weight_map = dict(zip(classes, weights))
        return np.array([weight_map[cls] for cls in y])
