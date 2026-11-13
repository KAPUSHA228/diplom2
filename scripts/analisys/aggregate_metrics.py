"""Сборка всех metrics.csv из папок запусков в единую таблицу."""

import json
from pathlib import Path

import pandas as pd

runs_dir = Path("../../experiments_output/runs")
rows = []

for run_dir in runs_dir.iterdir():
    if not run_dir.is_dir():
        continue
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        continue
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    df = pd.read_csv(metrics_path)
    df["balancer"] = cfg["balancer"]
    df["model"] = cfg["model"]
    df["dataset"] = cfg["dataset"]
    rows.append(df)

all_metrics = pd.concat(rows, ignore_index=True)
all_metrics.to_csv("../../experiments_output/summary/all_metrics.csv", index=False)
print(f"Собрано {len(all_metrics)} строк из {len(rows)} запусков")
print(all_metrics.groupby(["dataset", "model", "balancer"])["pr_auc"].mean().unstack())
