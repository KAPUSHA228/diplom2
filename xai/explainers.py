"""Обёртки над SHAP и LIME для табличных данных."""

import numpy as np
import shap

try:
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


def get_shap_values(model, X):
    """
    Возвращает матрицу SHAP-значений (n_samples, n_features) для класса 1.
    Поддерживает tree-модели (RF, XGB) и линейные (LR).
    """
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        if "scaler" in steps and "clf" in steps:
            X_scaled = steps["scaler"].transform(X)
            clf = steps["clf"]
            explainer = shap.LinearExplainer(clf, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            return np.asarray(shap_values)
        model = steps.get("clf", model)

    if hasattr(model, "get_booster") or hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.Explainer(model, X)

    shap_values = explainer.shap_values(X)
    # Приводим к одному массиву (класс 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    elif hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
    return np.asarray(shap_values)


def get_lime_values(model, X_train, X_explain, feature_names, instance_idx=0):
    """Локальное объяснение LIME для одного наблюдения. Возвращает dict {feature: weight}."""
    if not LIME_AVAILABLE:
        raise ImportError("lime не установлен. pip install lime")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=feature_names,
        class_names=["0", "1"],
        mode="classification",
        random_state=42,
    )
    exp = explainer.explain_instance(
        np.asarray(X_explain)[instance_idx],
        model.predict_proba,
        num_features=len(feature_names),
    )
    return dict(exp.as_list())
