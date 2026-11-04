"""Базовый интерфейс методов балансировки классов."""

from abc import ABC, abstractmethod
import numpy as np


class Balancer(ABC):
    """
    Абстрактный балансировщик.
    Контракт: fit_resample(X, y) -> (X_res, y_res).
    Для методов без ресемплинга (взвешивание классов) возвращает X, y без изменений,
    а веса доступны через get_sample_weight(y).
    """

    name: str = "base"

    @abstractmethod
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def get_sample_weight(self, y: np.ndarray):
        """По умолчанию веса не задаются."""
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
