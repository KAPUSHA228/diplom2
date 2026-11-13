"""
Оркестратор экспериментов: перебирает (балансировщик × модель × датасет),
включает повторы (n_repeats), сохраняет метрики и SHAP-значения для всех блоков.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from joblib import Parallel, delayed

from balancing.registry import get_balancer
from evaluation.metrics import compute_metrics
from experiments.grid import EXPERIMENT_GRID
from ml_core.core.config import config
from ml_core.core.data_registry import get_dataset
from ml_core.models.registry import get_model
from xai.explainers import get_shap_values


def _load_dataset(name: str):
    """Загружает датасет по имени из реестра."""
    meta = get_dataset(name)
    if meta.get("type") == "synthetic":
        params = meta["params"]
        X, y = make_classification(**params)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), pd.Series(
            y, name=meta.get("target", "target")
        )
    df = pd.read_csv(meta["path"])
    target = meta["target"]
    return df.drop(columns=[target]), df[target]


def _fit_model(model, X, y, sample_weight=None):
    """Обучает модель, корректно передавая sample_weight в Pipeline."""
    from sklearn.pipeline import Pipeline

    if sample_weight is None:
        model.fit(X, y)
        return model

    if isinstance(model, Pipeline):
        last_step = model.steps[-1][0]
        model.fit(X, y, **{f"{last_step}__sample_weight": sample_weight})
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def _run_id(balancer: str, model: str, dataset: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{ts}_{balancer}_{model}_{dataset}_{uuid.uuid4().hex[:6]}"


def _run_single_combination(ds_name, bal_name, model_name, grid, output_dir):
    """
    Выполняет все фолды для ОДНОЙ комбинации (датасет, балансировщик, модель).
    Возвращает словарь со сводными метриками.
    """
    # 1. Загружаем данные внутри процесса (безопасно для памяти)
    X, y = _load_dataset(ds_name)
    folds = grid["cv_folds"]
    n_repeats = grid.get("n_repeats", 1)

    run_dir = Path(output_dir) / "runs" / _run_id(bal_name, model_name, ds_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    all_shap_values = []

    for repeat in range(n_repeats):
        current_seed = 42 + repeat
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=current_seed)

        for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]

            balancer = get_balancer(bal_name)
            X_res, y_res = balancer.fit_resample(X_tr, y_tr)
            sample_weight = balancer.get_sample_weight(y_res) if bal_name == "class_weight" else None

            model = get_model(model_name)
            model = _fit_model(model, X_res, y_res, sample_weight=sample_weight)

            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

            m = compute_metrics(y_te, y_pred, y_proba)
            m["fold"] = fold_idx
            m["repeat"] = repeat
            m["seed"] = current_seed
            fold_metrics.append(m)

            shap_vals = get_shap_values(model, X_te)
            all_shap_values.append(shap_vals)

    # Сохранение результатов
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    if all_shap_values:
        # Выравниваем размерности (срезаем до минимальной), чтобы np.stack не падал
        min_len = min(a.shape[0] for a in all_shap_values)
        all_shap_values = [a[:min_len] for a in all_shap_values]
        np.save(run_dir / "shap_values.npy", np.stack(all_shap_values))

    config_json = {
        "balancer": bal_name,
        "model": model_name,
        "dataset": ds_name,
        "cv_folds": folds,
        "n_repeats": n_repeats,
        "seed_base": 42,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_json, f, ensure_ascii=False, indent=2)

    summary = metrics_df.drop(columns=["fold", "repeat", "seed"]).mean().to_dict()
    summary.update({"balancer": bal_name, "model": model_name, "dataset": ds_name, "run_dir": str(run_dir)})

    print(f"[OK] {bal_name} × {model_name} × {ds_name} → {run_dir.name}")
    return summary


class ExperimentRunner:
    def __init__(self, grid=None, output_dir=None):
        self.grid = grid or EXPERIMENT_GRID
        self.output_dir = Path(output_dir or config.EXPERIMENTS_OUTPUT_DIR)
        (self.output_dir / "runs").mkdir(parents=True, exist_ok=True)

    def run(self, datasets=None, balancers=None, models=None, n_jobs=1) -> pd.DataFrame:
        datasets = datasets or self.grid["dataset"]
        balancers = balancers or self.grid["balancer"]
        models = models or self.grid["model"]
        datasets = datasets or self.grid["dataset"]
        balancers = balancers or self.grid["balancer"]
        models = models or self.grid["model"]

        # Формируем список всех комбинаций
        combinations = list(itertools.product(datasets, balancers, models))

        print(f"Запуск {len(combinations)} комбинаций с параллелизмом (n_jobs={n_jobs})...")

        # Запускаем параллельно через joblib
        # verbose=10 будет печатать статус завершения задач
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_run_single_combination)(ds, bal, mod, self.grid, self.output_dir) for ds, bal, mod in combinations
        )

        return pd.DataFrame(results)
