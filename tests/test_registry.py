"""Проверка реестров балансировщиков и моделей."""

import pytest

from balancing.registry import BALANCERS, get_balancer
from ml_core.models.registry import MODEL_REGISTRY, get_model


def test_all_balancers_resolve():
    """Все ключи BALANCERS дают экземпляр."""
    for name in BALANCERS:
        balancer = get_balancer(name)
        assert hasattr(balancer, "fit_resample"), f"{name}: нет fit_resample"


def test_all_models_resolve():
    """Все ключи MODEL_REGISTRY дают экземпляр с fit/predict."""
    for name in MODEL_REGISTRY:
        model = get_model(name)
        assert hasattr(model, "fit"), f"{name}: нет fit"
        assert hasattr(model, "predict"), f"{name}: нет predict"


def test_expected_balancers_present():
    """Проверяем, что все 7 методов из плана зарегистрированы."""
    expected = {"baseline", "smote", "borderline_smote", "adasyn", "smote_tomek", "smote_enn", "class_weight"}
    assert expected.issubset(set(BALANCERS)), f"Не хватает: {expected - set(BALANCERS)}"


def test_expected_models_present():
    """Проверяем, что все 3 модели из плана зарегистрированы."""
    assert {"LR", "RF", "XGB"}.issubset(set(MODEL_REGISTRY))


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("nonexistent")
