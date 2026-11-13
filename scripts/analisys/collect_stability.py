"""
Сборка полных матриц стабильности по всем комбинациям.
Учитывает повторы (n_repeats) и считает корреляцию поблочно.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

runs_dir = Path("../../experiments_output/runs")
out_dir = Path("../../experiments_output/summary")
out_dir.mkdir(parents=True, exist_ok=True)

# Словарь для сбора данных: (датасет, модель) -> {метод: {блок: важность}}
collected_data = {}

for run_dir in runs_dir.iterdir():
    if not run_dir.is_dir():
        continue

    config_path = run_dir / "config.json"
    shap_path = run_dir / "shap_values.npy"

    if not config_path.exists() or not shap_path.exists():
        continue

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    ds_name = cfg["dataset"]
    model_name = cfg["model"]
    bal_name = cfg["balancer"]

    # Загружаем 3D-массив (15 блоков, N_тест, 20 признаков)
    shap_3d = np.load(shap_path)

    # Считаем важность признаков для каждого блока отдельно
    # np.abs(shap_3d) -> модуль, .mean(axis=1) -> усреднение по тестовым объектам
    # Получаем матрицу (15, 20)
    importance_matrix = np.abs(shap_3d).mean(axis=1)

    key = (ds_name, model_name)
    if key not in collected_data:
        collected_data[key] = {}
    if bal_name not in collected_data[key]:
        collected_data[key][bal_name] = {}

    # Сохраняем важности по блокам (индекс блока = fold + repeat * cv_folds)
    # В нашем случае 5 фолдов * 3 повтора = 15, индексы от 0 до 14
    for i in range(importance_matrix.shape[0]):
        collected_data[key][bal_name][i] = importance_matrix[i]

# Теперь считаем матрицы стабильности
for (ds_name, model_name), balancers_data in collected_data.items():
    balancers = list(balancers_data.keys())
    n_balancers = len(balancers)

    # Инициализируем матрицы корреляций и top-k
    corr_matrix = np.zeros((n_balancers, n_balancers))
    overlap_matrix = np.zeros((n_balancers, n_balancers))

    for i, bal_a in enumerate(balancers):
        for j, bal_b in enumerate(balancers):
            if i == j:
                corr_matrix[i, j] = 1.0
                overlap_matrix[i, j] = 1.0
                continue

            # Считаем корреляцию для каждого из 15 блоков
            block_corrs = []
            block_overlaps = []

            for block_idx in range(len(balancers_data[bal_a])):
                vec_a = balancers_data[bal_a][block_idx]
                vec_b = balancers_data[bal_b][block_idx]

                # Корреляция Спирмена
                corr, _ = spearmanr(vec_a, vec_b)
                block_corrs.append(corr)

                # Top-K (например, K=5)
                top_a = set(np.argsort(-vec_a)[:5])
                top_b = set(np.argsort(-vec_b)[:5])
                block_overlaps.append(len(top_a & top_b) / 5.0)

            # Усредняем по 15 блокам
            corr_matrix[i, j] = np.mean(block_corrs)
            overlap_matrix[i, j] = np.mean(block_overlaps)

    # Сохраняем в CSV
    prefix = f"{ds_name}_{model_name}"
    pd.DataFrame(corr_matrix, index=balancers, columns=balancers).to_csv(out_dir / f"stability_corr_{prefix}.csv")
    pd.DataFrame(overlap_matrix, index=balancers, columns=balancers).to_csv(out_dir / f"stability_overlap_{prefix}.csv")
    print(f"[OK] Собраны матрицы для {prefix}")

print("\nГотово. Матрицы в experiments_output/summary/")
