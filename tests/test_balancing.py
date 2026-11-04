"""Проверка корректности методов балансировки."""

import numpy as np
import pytest

from balancing.registry import get_balancer


@pytest.mark.parametrize(
    "name,max_diff",
    [
        ("smote", 0.0),
        ("borderline_smote", 0.0),
        ("adasyn", 0.05),  # адаптивный, допускаем 5%
        ("smote_tomek", 0.0),
        ("smote_enn", 0.25),  # удаляет шумные объекты, допускаем 25%
    ],
)
def test_balancer_equilibrates_classes(name, max_diff, imbalanced_data):
    """После ресемплинга классы должны быть сбалансированы (равное число)."""
    X, y = imbalanced_data
    X_res, y_res = get_balancer(name).fit_resample(X, y)
    counts = np.bincount(y_res)
    diff_ratio = abs(counts[0] - counts[1]) / counts.min()
    assert diff_ratio <= max_diff, f"{name}: {counts}, расхождение {diff_ratio:.1%} > {max_diff:.0%}"


def test_baseline_does_not_change_data(imbalanced_data):
    """Базовый вариант не меняет данные."""
    X, y = imbalanced_data
    X_res, y_res = get_balancer("baseline").fit_resample(X, y)
    assert np.array_equal(X, X_res)
    assert np.array_equal(y, y_res)


def test_class_weight_returns_weights(imbalanced_data):
    """Взвешивание классов не ресемплит, но даёт веса, обратные частотам."""
    X, y = imbalanced_data
    balancer = get_balancer("class_weight")
    X_res, y_res = balancer.fit_resample(X, y)
    assert np.array_equal(X, X_res)
    weights = balancer.get_sample_weight(y)
    assert weights is not None
    assert len(weights) == len(y)
    # Вес миноритарного класса должен быть больше
    assert weights[y == 1].mean() > weights[y == 0].mean()


def test_unknown_balancer_raises():
    """Неизвестный балансировщик → KeyError."""
    with pytest.raises(KeyError):
        get_balancer("nonexistent")
