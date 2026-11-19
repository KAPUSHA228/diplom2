import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# Загружаем агрегированные метрики (по 15 разбиениям)
df = pd.read_csv("../../experiments_output/summary/all_metrics.csv")
df = df[df["dataset"].str.match(r"^synth_\d+_r\d$")]

# Сначала усредняем 15 разбиений внутри каждого набора
grouped = df.groupby(["dataset", "model", "balancer"])[["pr_auc", "mcc", "f2", "g_mean"]].mean().reset_index()

metrics_to_test = ["pr_auc", "mcc", "f2", "g_mean"]
stats_rows = []

for model in ["LR", "RF", "XGB"]:
    for metric in metrics_to_test:
        sub = grouped[grouped["model"] == model]

        # Нам нужно сравнить baseline с SMOTE и baseline с Class Weight
        # Создаем сводную таблицу: строки - датасеты, столбцы - методы
        pivot = sub.pivot(index="dataset", columns="balancer", values=metric)

        if "baseline" in pivot.columns:
            labels = {"smote": "SMOTE", "class_weight": "CW"}
            for col in pivot.columns:
                if col == "baseline":
                    continue
                _, p = wilcoxon(pivot[col], pivot["baseline"])
                stats_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "comp": labels.get(col, col),
                        "delta": (pivot[col] - pivot["baseline"]).mean(),
                        "p": p,
                    }
                )
# Применяем поправку Холма
df_stats = pd.DataFrame(stats_rows)
df_stats["p_holm"] = df_stats.groupby("metric")["p"].transform(lambda x: multipletests(x, method="holm")[1])

print("\n=== СТАТИСТИКА ПО МЕТРИКАМ КАЧЕСТВА (по 15 наборам) ===")
print(df_stats[df_stats["p_holm"] < 0.05].to_string(index=False))
