"""
Модуль для оценки моделей и объяснения предсказаний
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml_core.core.config import config
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from artifacts.archive.error_handler import logger


def generate_shap_explanations(
    model,
    X: pd.DataFrame,
    feature_names: list,
    threshold: float = 0.5,
    top_n: int = 5,
    target_name: str = "Целевая переменная",
    n_students: int = 5,
):
    """
    Генерирует SHAP-объяснения для первых 5 наблюдений.

    Args:
        model: обученная модель (XGBoost/RF/LogisticRegression)
        X: данные для объяснения (DataFrame)
        feature_names: список имён признаков
        threshold: порог классификации для определения уровня
        top_n: число объяснений для возврата
        target_name: имя целевой переменной для текстовых объяснений
        n_students: количество студентов для объяснения (по умолчанию 5)

    Returns:
        list[dict]: [{'student_index', 'risk_probability', 'risk_level', 'explanation'}]
    """
    try:
        # 1. Получаем вероятности риска
        proba = model.predict_proba(X)[:, 1]

        # 2. Берём топ-N самых рискованных студентов
        n_to_show = min(n_students, len(X))
        top_indices = np.argsort(proba)[-n_to_show:][::-1]

        logger.info(f"SHAP: выбрано {len(top_indices)} студентов из {len(X)} (топ по риску)")

        # Единый способ создания explainer
        if isinstance(model, (XGBClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X) if hasattr(model, "coef_") else shap.Explainer(model)

            # 4. Вычисляем SHAP значения ТОЛЬКО для выбранных студентов (экономия времени!)
        X_selected = X.iloc[top_indices]
        shap_values = explainer.shap_values(X_selected)

        # 5. Приведение к единому формату (для класса 1)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]

        # 6. Формируем объяснения
        explanations = []
        for idx, (orig_idx, shap_row) in enumerate(zip(top_indices, shap_values)):
            risk_prob = proba[orig_idx]

            # Сортируем признаки по влиянию
            feature_effects = list(zip(feature_names, shap_row))
            feature_effects.sort(key=lambda x: abs(x[1]), reverse=True)

            explanation = {
                "student_index": int(orig_idx),
                "risk_probability": float(risk_prob),
                "risk_level": "высокое" if risk_prob > threshold else "низкое",
                "top_features": [
                    {
                        "feature": feat,
                        "shap_value": float(val),
                        "effect": f"повышает {target_name}" if val > 0 else f"понижает {target_name}",
                    }
                    for feat, val in feature_effects[:top_n]
                ],
            }
            explanation["explanation"] = generate_text_explanation(explanation, target_name)
            explanations.append(explanation)

        return explanations

    except Exception as e:
        logger.error(f"SHAP generation failed: {str(e)}", exc_info=True)
        return []


def generate_text_explanation(exp, target_name="Целевая переменная"):
    """
    Преобразует dict SHAP-объяснения в читаемый текст.

    Args:
        exp: dict с ключами 'risk_probability', 'risk_level', 'factors'
        target_name: имя целевой переменной

    Returns:
        str: текстовое объяснение
    """
    text = f"Вероятность {target_name}: {exp['risk_probability']:.1%} ({exp['risk_level']})\n\n"
    text += "Основные факторы:\n"

    for feat in exp["top_features"]:
        # Определяем направление по значению SHAP
        arrow = "↑" if feat.get("shap_value", 0) > 0 else "↓"
        text += f"• {feat['feature']}: {arrow} ({feat['shap_value']:.3f})\n"

    return text


# ---- SHAP анализ ----
def explain_model_with_shap(model, X_explain):
    """
    Генерирует summary plot SHAP и текстовые объяснения.

    Args:
        model: обученная модель (XGBoost/RF/LogisticRegression)
        X_explain: данные для объяснения (DataFrame)

    Returns:
        list[str]: текстовые объяснения для каждого наблюдения
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_explain)
    shap.summary_plot(shap_values, X_explain, show=False)
    # Форматировать текстовое объяснение для каждого объекта
    explanations = []
    for i in range(X_explain.shape[0]):
        vals = shap_values[i].values
        # Преобразуем в одномерный массив, если нужно
        if vals.ndim > 1:
            vals = vals.flatten()
        top_idx = np.argsort(np.abs(vals))[-2:][::-1]
        top_feats = [X_explain.columns[idx] for idx in top_idx]
        top_vals = [float(vals[idx]) for idx in top_idx]
        txt = f"Риск из-за: {top_feats[0]} ({top_vals[0]:+.2f}), {top_feats[1]} ({top_vals[1]:+.2f})"
        explanations.append(txt)
    return explanations


def generate_detailed_explanation(model, X_explain, student_idx, feature_names, threshold=0.5):
    """
    Генерирует детальное объяснение для конкретного студента.

    Args:
        model: обученная модель
        X_explain: DataFrame с данными студентов
        student_idx: индекс студента в DataFrame
        feature_names: список имён признаков
        threshold: порог классификации риска

    Returns:
        str: детальное текстовое объяснение с рекомендациями
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_explain.iloc[[student_idx]])

    # Получаем предсказание
    proba = model.predict_proba(X_explain.iloc[[student_idx]])[0, 1]
    risk_level = "высокий" if proba >= threshold else "низкий"

    # Извлекаем SHAP значения
    vals = shap_values.values[0]
    if vals.ndim > 1:
        vals = vals.flatten()

    # Убеждаемся, что количество значений совпадает с количеством признаков
    n_features = len(feature_names)
    if len(vals) != n_features:
        # Берем только первые n_features значений
        vals = vals[:n_features]

    # Топ-3 фактора (по абсолютному значению), но не больше доступных признаков
    top_n = min(3, n_features)
    top_3_idx = np.argsort(np.abs(vals))[-top_n:][::-1]
    # Фильтруем индексы, чтобы они были в допустимых пределах
    top_3_idx = [idx for idx in top_3_idx if 0 <= idx < n_features]
    top_features = [feature_names[idx] for idx in top_3_idx]
    top_shap_vals = [float(vals[idx]) for idx in top_3_idx]

    # Формируем объяснение
    explanation_parts = [
        f"Вероятность академического риска: {proba:.1%} (уровень: {risk_level})",
        "\nОсновные факторы риска:",
    ]

    for feat, shap_val in zip(top_features, top_shap_vals):
        direction = "увеличивает" if shap_val > 0 else "снижает"
        explanation_parts.append(f"  • {feat}: {direction} риск (вклад: {shap_val:+.3f})")

    # Простые рекомендации на основе признаков
    recommendations = []
    for feat, shap_val in zip(top_features, top_shap_vals):
        if shap_val > 0:  # Фактор увеличивает риск
            if "trend" in feat.lower() or "grade" in feat.lower():
                recommendations.append("Рекомендуется обратить внимание на динамику успеваемости")
            elif "attendance" in feat.lower():
                recommendations.append("Необходимо улучшить посещаемость занятий")
            elif "stress" in feat.lower() or "cognitive" in feat.lower():
                recommendations.append("Требуется снижение когнитивной нагрузки и уровня стресса")

    if recommendations:
        explanation_parts.append("\nРекомендации:")
        explanation_parts.extend([f"  • {rec}" for rec in recommendations[:3]])

    return "\n".join(explanation_parts)


def generate_batch_explanations(model, X_explain, feature_names, threshold=0.5, top_n=10):
    """
    Генерирует объяснения для top-N студентов с наибольшим риском.

    Args:
        model: обученная модель
        X_explain: DataFrame с данными студентов
        feature_names: список имён признаков
        threshold: порог классификации риска
        top_n: число студентов с наибольшим риском

    Returns:
        list[dict]: список объяснений для каждого студента
    """
    probas = model.predict_proba(X_explain)[:, 1]
    top_risk_indices = np.argsort(probas)[-top_n:][::-1]

    explanations_list = []
    for idx in top_risk_indices:
        explanation = generate_detailed_explanation(model, X_explain, idx, feature_names, threshold)
        explanations_list.append(
            {"student_index": int(idx), "risk_probability": float(probas[idx]), "explanation": explanation}
        )

    return explanations_list


# ---- Визуализация метрик и важности признаков ----
def plot_feature_importance_png(model, feature_names, name="model", output_path=None, top_n=10):
    """
    Строит горизонтальную bar chart важности признаков (matplotlib).

    Args:
        model: модель с attribute feature_importances_ (RF/XGB)
        feature_names: список имён признаков
        name: имя модели (для имени файла)
        output_path: путь сохранения (по умолчанию: logs/fi_{name}.png)
        top_n: сколько лучших признаков показать

    Returns:
        None (сохраняет PNG)
    """
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(6, 4))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title(f"Feature importance: {name}")
    plt.tight_layout()
    if output_path is None:
        output_path = config.LOGS_DIR / f"fi_{name}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
