"""
Метрики стабильности SHAP-объяснений при смене метода балансировки.
Ранговая корреляция Спирмена, доля совпадений в топ-K, бутстрэп-дисперсия.
"""

import numpy as np
from scipy.stats import spearmanr

from xai.explainers import get_shap_values


def feature_importance(shap_values: np.ndarray) -> np.ndarray:
    """Глобальная важность признака = среднее |SHAP| по наблюдениям."""
    return np.abs(shap_values).mean(axis=0)


def rank_correlation(shap_1: np.ndarray, shap_2: np.ndarray) -> float:
    """
    Ранговая корреляция Спирмена между важностями признаков двух балансировщиков.
    Возвращает значение в [-1, 1]; 1 — идеальное совпадение ранжирований.
    """
    imp_1 = feature_importance(shap_1)
    imp_2 = feature_importance(shap_2)
    if np.allclose(imp_1, imp_1[0]) or np.allclose(imp_2, imp_2[0]):
        return 0.0
    corr, _ = spearmanr(imp_1, imp_2)
    return float(corr) if not np.isnan(corr) else 0.0


def top_k_overlap(shap_1: np.ndarray, shap_2: np.ndarray, k: int = 5) -> float:
    """
    Доля совпавших признаков в топ-K по важности.
    Возвращает значение в [0, 1]; 1 — топ-K полностью совпал.
    """
    imp_1 = feature_importance(shap_1)
    imp_2 = feature_importance(shap_2)
    top_1 = set(np.argsort(-imp_1)[:k])
    top_2 = set(np.argsort(-imp_2)[:k])
    return len(top_1 & top_2) / k


def bootstrap_shap_variance(model, X, n_bootstrap: int = 30, random_state: int = 42) -> np.ndarray:
    """
    Дисперсия SHAP-значений при бутстрэпе выборок.
    Возвращает массив (n_features,) — дисперсию важности каждого признака.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    importances = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_boot = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        shap_vals = get_shap_values(model, X_boot)
        importances.append(feature_importance(shap_vals))
    return np.var(np.vstack(importances), axis=0)


def stability_matrix(shap_dict: dict, k: int = 5) -> dict:
    """
    Матрица стабильности между всеми парами балансировщиков.
    shap_dict: {balancer_name: shap_values}
    Возвращает {"rank_correlation": DataFrame, "top_k_overlap": DataFrame}.
    """
    import pandas as pd

    names = list(shap_dict.keys())
    corr = pd.DataFrame(index=names, columns=names, dtype=float)
    overlap = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            corr.loc[a, b] = rank_correlation(shap_dict[a], shap_dict[b])
            overlap.loc[a, b] = top_k_overlap(shap_dict[a], shap_dict[b], k=k)
    return {"rank_correlation": corr, "top_k_overlap": overlap}
