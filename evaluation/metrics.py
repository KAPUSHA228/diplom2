"""Метрики качества, устойчивые к дисбалансу классов."""

import numpy as np
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
)


def geometric_mean(y_true, y_pred):
    """Среднее геометрическое чувствительности и специфичности (G-Mean)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(sensitivity * specificity))


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Основные метрики для задачи с дисбалансом.
    PR-AUC — первичная метрика (устойчива к сильному дисбалансу).
    """
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "g_mean": geometric_mean(y_true, y_pred),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Вычисляет метрики классификации: F1, Precision, Recall, ROC-AUC.

    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        y_proba: вероятности положительного класса (для ROC-AUC)

    Returns:
        dict: {'f1', 'precision', 'recall', 'roc_auc' (если y_proba есть)}
    """
    metrics = {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


# def plot_roc_curves(models_dict, X_test, y_test):
#     """
#     Строит ROC-кривые для нескольких моделей (Plotly).
#
#     Args:
#         models_dict: dict {name: model} обученных моделей
#         X_test: тестовые признаки
#         y_test: истинные тестовые метки
#
#     Returns:
#         plotly.graph_objects.Figure: ROC-кривые всех моделей
#     """
#     fig = go.Figure()
#
#     # Диагональная линия (случайная модель)
#     fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
#
#     for name, model in models_dict.items():
#         if hasattr(model, "predict_proba"):
#             y_proba = model.predict_proba(X_test)[:, 1]
#             fpr, tpr, _ = roc_curve(y_test, y_proba)
#             auc = roc_auc_score(y_test, y_proba)
#
#             fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
#
#     fig.update_layout(
#         title="ROC-кривые моделей",
#         xaxis_title="False Positive Rate",
#         yaxis_title="True Positive Rate",
#         width=600,
#         height=500,
#     )
#
#     return fig
#
#
# def plot_confusion_matrix(y_true, y_pred, model_name):
#     """
#     Строит confusion matrix (Plotly).
#
#     Args:
#         y_true: истинные метки
#         y_pred: предсказанные метки
#         model_name: имя модели (для заголовка)
#
#     Returns:
#         plotly.graph_objects.Figure: confusion matrix
#     """
#     cm = confusion_matrix(y_true, y_pred)
#
#     fig = px.imshow(
#         cm,
#         text_auto=True,
#         aspect="auto",
#         labels=dict(x="Predicted", y="True"),
#         title=f"Confusion Matrix: {model_name}",
#         color_continuous_scale="Blues",
#     )
#
#     return fig
#
#
# def plot_feature_importance(model, feature_names, top_n=10):
#     """
#     Строит bar chart важности признаков (Plotly).
#
#     Args:
#         model: обученная модель с feature_importances_ или coef_
#         feature_names: список имён признаков
#         top_n: сколько лучших признаков показать
#
#     Returns:
#         plotly.graph_objects.Figure: bar chart важности
#     """
#     if hasattr(model, "feature_importances_"):
#         importances = model.feature_importances_
#     elif hasattr(model, "coef_"):
#         importances = np.abs(model.coef_[0])
#     else:
#         return None
#
#     # Сортируем
#     indices = np.argsort(importances)[-top_n:]
#
#     fig = px.bar(
#         x=importances[indices],
#         y=[feature_names[i] for i in indices],
#         orientation="h",
#         title="Важность признаков",
#         labels={"x": "Importance", "y": ""},
#     )
#
#     return fig
