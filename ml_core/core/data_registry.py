"""
Реестр датасетов для воспроизводимых экспериментов.
Каждый датасет описан метаданными: путь, целевая колонка, доля миноритарного класса, источник.
"""

from typing import Any, Dict

from ml_core.core.config import config

DATASETS: Dict[str, Dict[str, Any]] = {
    "uci_student_dropout": {
        "path": str(config.BASE_DIR / "data" / "uci_student_dropout_prepared.csv"),
        "target": "risk_flag",
        "minority_ratio": None,
        "source": "UCI ML Repository (ID 697)",
        "type": "real",
    }
}
# Программно генерируем 15 синтетических наборов (3 варианта на каждую долю)
for ratio in [1, 5, 10, 20, 30]:
    for r_idx in range(1, 4):  # r1, r2, r3
        # random_state меняем для каждого набора, чтобы данные были разными
        seed = 42 + r_idx - 1
        name = f"synth_{ratio:02d}_r{r_idx}"

        DATASETS[name] = {
            "generator": "make_classification",
            "params": {
                "n_samples": 2000,
                "n_features": 20,
                "n_informative": 5,
                "weights": [1 - (ratio / 100), ratio / 100],
                "random_state": seed,
                "shuffle": False,
                "flip_y": 0,
            },
            "target": "target",
            "minority_ratio": ratio / 100,
            "source": "sklearn.datasets",
            "type": "synthetic",
        }

# Специальный тестовый набор для проверки LR (без избыточных признаков)
DATASETS["synth_lr_test"] = {
    "generator": "make_classification",
    "params": {
        "n_samples": 2000,
        "n_features": 20,
        "n_informative": 5,
        "n_redundant": 0,
        "weights": [0.9, 0.1],
        "random_state": 42,
        "shuffle": False,
        "flip_y": 0,
    },
    "target": "target",
    "minority_ratio": 0.1,
    "source": "sklearn.datasets",
    "type": "synthetic",
}


def get_dataset(name: str) -> Dict[str, Any]:
    """Возвращает описание датасета по имени. Бросает KeyError, если нет."""
    if name not in DATASETS:
        raise KeyError(f"Датасет '{name}' не зарегистрирован. Доступны: {list(DATASETS)}")
    return DATASETS[name]


def list_datasets() -> list:
    """Список имён всех зарегистрированных датасетов."""
    return list(DATASETS.keys())
