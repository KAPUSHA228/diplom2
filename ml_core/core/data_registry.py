"""
Реестр датасетов для воспроизводимых экспериментов.
Каждый датасет описан метаданными: путь, целевая колонка, доля миноритарного класса, источник.
"""

from typing import Any, Dict

from ml_core.core.config import config

DATASETS: Dict[str, Dict[str, Any]] = {
    # Ранее сгенерированный сложноструктурный набор (не настоящие данные, но сложнее стандартной синтетики)
    "student_autumn": {
        "path": str(config.ANALYSIS_DATA_DIR / "synthetic" / "students_autumn_2023.csv"),
        "target": "risk_flag",
        "minority_ratio": 0.116,
        "source": "ранее сгенерированный (complex)",
        "type": "real_like",  # В коде runner.py это обрабатывается как pd.read_csv
    },
    # Синтетические наборы с разной долей дисбаланса (от 1% до 30%)
    "synth_01": {
        "generator": "make_classification",
        "params": {"n_samples": 2000, "n_features": 20, "weights": [0.99, 0.01], "random_state": 42},
        "target": "target",
        "minority_ratio": 0.01,
        "source": "sklearn.datasets",
        "type": "synthetic",
    },
    "synth_05": {
        "generator": "make_classification",
        "params": {"n_samples": 2000, "n_features": 20, "weights": [0.95, 0.05], "random_state": 42},
        "target": "target",
        "minority_ratio": 0.05,
        "source": "sklearn.datasets",
        "type": "synthetic",
    },
    "synth_10": {
        "generator": "make_classification",
        "params": {"n_samples": 2000, "n_features": 20, "weights": [0.90, 0.10], "random_state": 42},
        "target": "target",
        "minority_ratio": 0.10,
        "source": "sklearn.datasets",
        "type": "synthetic",
    },
    "synth_20": {
        "generator": "make_classification",
        "params": {"n_samples": 2000, "n_features": 20, "weights": [0.80, 0.20], "random_state": 42},
        "target": "target",
        "minority_ratio": 0.20,
        "source": "sklearn.datasets",
        "type": "synthetic",
    },
    "synth_30": {
        "generator": "make_classification",
        "params": {"n_samples": 2000, "n_features": 20, "weights": [0.70, 0.30], "random_state": 42},
        "target": "target",
        "minority_ratio": 0.30,
        "source": "sklearn.datasets",
        "type": "synthetic",
    },
}


def get_dataset(name: str) -> Dict[str, Any]:
    """Возвращает описание датасета по имени. Бросает KeyError, если нет."""
    if name not in DATASETS:
        raise KeyError(f"Датасет '{name}' не зарегистрирован. Доступны: {list(DATASETS)}")
    return DATASETS[name]


def list_datasets() -> list:
    """Список имён всех зарегистрированных датасетов."""
    return list(DATASETS.keys())
