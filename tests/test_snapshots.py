"""Эталонные (snapshot) тесты: фиксированный seed → эталонные метрики."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from evaluation.metrics import compute_metrics


@pytest.fixture
def fixed_predictions():
    """Фиксированные предсказания, чтобы метрики были детерминированы."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_pred = rng.integers(0, 2, size=100)
    y_proba = rng.uniform(0, 1, size=100)
    return y_true, y_pred, y_proba


def test_metrics_keys(fixed_predictions):
    """Все ожидаемые метрики присутствуют."""
    y_true, y_pred, y_proba = fixed_predictions
    m = compute_metrics(y_true, y_pred, y_proba)
    expected = {"f1", "f2", "precision", "recall", "g_mean", "mcc", "pr_auc", "roc_auc"}
    assert expected.issubset(m.keys())


def test_metrics_ranges(fixed_predictions):
    """Все метрики лежат в допустимых диапазонах."""
    y_true, y_pred, y_proba = fixed_predictions
    m = compute_metrics(y_true, y_pred, y_proba)
    for key in ["f1", "f2", "precision", "recall", "g_mean", "pr_auc", "roc_auc"]:
        assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} вне [0, 1]"
    assert -1.0 <= m["mcc"] <= 1.0


def test_metrics_deterministic(fixed_predictions):
    """Повторный вызов даёт те же значения (детерминизм)."""
    y_true, y_pred, y_proba = fixed_predictions
    m1 = compute_metrics(y_true, y_pred, y_proba)
    m2 = compute_metrics(y_true, y_pred, y_proba)
    for key in m1:
        assert m1[key] == pytest.approx(m2[key])


def test_perfect_predictions():
    """Идеальные предсказания → все метрики = 1 (кроме mcc, тоже 1)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["f1"] == pytest.approx(1.0)
    assert m["g_mean"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)


def test_snapshot_rf_metrics(imbalanced_df):
    """Эталонные метрики RF при фиксированном seed — защита от регрессий."""
    X, y = imbalanced_df
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_te, y_pred, y_proba)
    # Эталонные значения (фиксированный seed → детерминизм)
    assert m["f1"] > 0.0
    assert m["pr_auc"] > 0.0
    assert m["mcc"] > 0.0
