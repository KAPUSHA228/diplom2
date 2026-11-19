"""Сравнение SHAP-объяснений между балансировщиками."""

from pathlib import Path

import numpy as np

from xai.stability import stability_matrix


def load_shap_from_runs(runs_dir, dataset: str, model: str) -> dict:
    """
    Собирает SHAP-значения из папок запусков.
    Возвращает {balancer: shap_values} для заданных датасета и модели.
    """
    shap_dict = {}
    runs_dir = Path(runs_dir)
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        shap_path = run_dir / "shap_values.npy"
        config_path = run_dir / "config.json"
        if not shap_path.exists() or not config_path.exists():
            continue
        import json

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("dataset") == dataset and cfg.get("model") == model:
            shap_dict[cfg["balancer"]] = np.load(shap_path)
    return shap_dict


def compare_shap_across_balancers(shap_dict: dict, k: int = 5) -> dict:
    """Матрицы ранговой корреляции и доли совпадений в топ-K между балансировщиками."""
    return stability_matrix(shap_dict, k=k)


def save_stability_tables(shap_dict: dict, output_dir, k: int = 5):
    """Сохраняет матрицы стабильности в CSV и возвращает пути."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mats = stability_matrix(shap_dict, k=k)
    paths = {}
    for name, df in mats.items():
        path = output_dir / f"shap_{name}.csv"
        df.to_csv(path, encoding="utf-8")
        paths[name] = str(path)
    return paths
