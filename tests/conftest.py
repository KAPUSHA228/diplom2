"""Общие фикстуры для тестов."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def imbalanced_data():
    """Несбалансированная выборка: 90% класса 0, 10% класса 1."""
    rng = np.random.default_rng(42)
    n_majority, n_minority = 900, 100
    X_major = rng.normal(0, 1, size=(n_majority, 5))
    X_minor = rng.normal(1, 1, size=(n_minority, 5))
    X = np.vstack([X_major, X_minor])
    y = np.array([0] * n_majority + [1] * n_minority)
    return X, y


@pytest.fixture
def imbalanced_df():
    """То же, но в виде DataFrame — для моделей, ожидающих имена колонок."""
    rng = np.random.default_rng(42)
    n_majority, n_minority = 900, 100
    X_major = rng.normal(0, 1, size=(n_majority, 5))
    X_minor = rng.normal(1, 1, size=(n_minority, 5))
    X = np.vstack([X_major, X_minor])
    y = np.array([0] * n_majority + [1] * n_minority)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    return df, pd.Series(y, name="target")


@pytest.fixture
def trained_rf(imbalanced_df):
    """Обученный RandomForest на несбалансированных данных."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = imbalanced_df
    model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
    model.fit(X, y)
    return model, X
