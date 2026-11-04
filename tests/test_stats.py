"""Проверка статистических функций (тесты на основе свойств)."""

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st

from experiments.stats import bonferroni_correction, compare_balancers, confidence_intervals


def _make_results_df():
    """Синтетическая сводная таблица: 2 балансировщика × 2 модели × 5 фолдов."""
    rows = []
    rng = np.random.default_rng(42)
    for bal in ["baseline", "smote"]:
        for model in ["LR", "RF"]:
            for fold in range(5):
                rows.append(
                    {
                        "balancer": bal,
                        "model": model,
                        "fold": fold,
                        "pr_auc": float(rng.uniform(0.3, 0.9)),
                    }
                )
    return pd.DataFrame(rows)


@given(st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=20))
def test_bonferroni_range(p_values):
    """После поправки Бонферрони p-value остаётся в [0, 1]."""
    corrected = bonferroni_correction(p_values)
    assert np.all(corrected >= 0)
    assert np.all(corrected <= 1.0)


def test_bonferroni_monotone():
    """Поправка не уменьшает p-value."""
    p = np.array([0.01, 0.02, 0.5])
    corrected = bonferroni_correction(p)
    assert np.all(corrected >= p)


def test_bonferroni_formula():
    """p * m, с ограничением 1.0."""
    p = np.array([0.01, 0.02])
    corrected = bonferroni_correction(p)
    np.testing.assert_allclose(corrected, [0.02, 0.04])


def test_compare_balancers_returns_pvalues_in_range():
    """Wilcoxon даёт p-value в [0, 1], пары не повторяются."""
    df = _make_results_df()
    result = compare_balancers(df, metric="pr_auc", by="model")
    assert not result.empty
    assert np.all(result["p_value"] >= 0)
    assert np.all(result["p_value"] <= 1.0)
    # пары (a, b) уникальны внутри модели
    for model, group in result.groupby("model"):
        pairs = set(zip(group["a"], group["b"]))
        assert len(pairs) == len(group)


def test_confidence_intervals_bounds():
    """Нижняя граница <= среднее <= верхняя граница."""
    df = _make_results_df()
    ci = confidence_intervals(df, metric="pr_auc")
    assert not ci.empty
    assert np.all(ci["ci_low"] <= ci["mean"])
    assert np.all(ci["mean"] <= ci["ci_high"])
