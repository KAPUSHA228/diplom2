import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# Загружаем данные из нашего нового файла
df = pd.read_csv("../../experiments_output/summary/per_split_metrics.csv")

# Оставляем только синтетику
df = df[df["dataset"].str.match(r"^synth_\d+_r\d$")]

stats_rows = []
for model in ["LR", "RF", "XGB"]:
    # Берем только SMOTE и CW
    sub = df[(df["model"] == model) & (df["balancer"].isin(["smote", "class_weight"]))]

    # Группируем по набору (датасет) и усредняем по 15 разбиениям
    # Получаем 15 значений для SMOTE и 15 для CW (5 долей * 3 генерации)
    grouped = sub.groupby(["dataset", "balancer"])["spearman"].mean().unstack()

    if "smote" in grouped.columns and "class_weight" in grouped.columns:
        # Считаем дельту (Class Weight - SMOTE) по 15 наборам
        delta = grouped["class_weight"] - grouped["smote"]

        # Критерий Уилкоксона по 15 парам
        stat, p = wilcoxon(grouped["class_weight"], grouped["smote"])

        stats_rows.append({"model": model, "mean_delta": delta.mean(), "p_value": p})

# Применяем поправку Холма на 3 модели
df_stats = pd.DataFrame(stats_rows)
df_stats["p_holm"] = multipletests(df_stats["p_value"], method="holm")[1]
print("\n=== ПРАВИЛЬНАЯ СТАТИСТИКА (по 15 независимым наборам) ===")
print(df_stats.to_string(index=False))
