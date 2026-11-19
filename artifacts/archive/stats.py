"""Статистическая обработка результатов: Wilcoxon, поправка Бонферрони, доверительные интервалы."""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def compare_balancers(results_df: pd.DataFrame, metric: str = "pr_auc", by: str = "model") -> pd.DataFrame:
    """
    Парные сравнения Wilcoxon между всеми парами балансировщиков (внутри каждой модели).
    results_df: сводная таблица с колонками balancer, model, <metric>.
    """
    rows = []
    for model_name, group in results_df.groupby(by):
        pairs = itertools.combinations(group["balancer"].unique(), 2)
        for a, b in pairs:
            va = group[group["balancer"] == a].sort_values("fold")[metric].values
            vb = group[group["balancer"] == b].sort_values("fold")[metric].values
            n = min(len(va), len(vb))
            if n < 2:
                continue
            try:
                stat, p = wilcoxon(va[:n], vb[:n])
            except ValueError:
                # все разности нулевые — Wilcoxon не определён
                continue
            rows.append(
                {"model": model_name, "a": a, "b": b, "metric": metric, "stat": float(stat), "p_value": float(p)}
            )
    return pd.DataFrame(rows)


def bonferroni_correction(p_values) -> np.ndarray:
    """Поправка Бонферрони: p * m, с ограничением сверху 1.0."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    return np.minimum(p * m, 1.0)


def confidence_intervals(results_df: pd.DataFrame, metric: str = "pr_auc", alpha: float = 0.05) -> pd.DataFrame:
    """Доверительные интервалы среднего по фолдам (нормальное приближение)."""
    rows = []
    for (bal, model), group in results_df.groupby(["balancer", "model"]):
        vals = group[metric].values
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        z = 1.96 if alpha == 0.05 else 2.576
        rows.append({"balancer": bal, "model": model, "mean": mean, "ci_low": mean - z * se, "ci_high": mean + z * se})
    return pd.DataFrame(rows)
