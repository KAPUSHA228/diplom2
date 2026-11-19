import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../../experiments_output/summary/all_metrics.csv")
df = df[df["dataset"].str.match(r"^synth_\d+_r\d$")]

# Усредняем по 15 наборам
grouped = df.groupby(["dataset", "model", "balancer"])[["pr_auc", "mcc", "f2", "g_mean"]].mean().reset_index()

# Считаем разницу с baseline для каждого набора
# Сначала создаем таблицу baseline
baseline = grouped[grouped["balancer"] == "baseline"].set_index(["dataset", "model"])[["pr_auc", "mcc", "f2", "g_mean"]]
baseline.columns = [f"{c}_baseline" for c in baseline.columns]

# Теперь приклеиваем к каждому методу
grouped = grouped.join(baseline, on=["dataset", "model"])

# Считаем дельты
for metric in ["pr_auc", "mcc", "f2", "g_mean"]:
    grouped[f"{metric}_delta"] = grouped[metric] - grouped[f"{metric}_baseline"]

# Берем только SMOTE и Class Weight для наглядности (или все методы)
methods = ["class_weight", "smote"]
grouped = grouped[grouped["balancer"].isin(methods)]

# Строим график (средние дельты по моделям)
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))

# Формируем данные для графика
plot_data = []
for method in methods:
    for metric in ["pr_auc", "mcc", "f2", "g_mean"]:
        mean_delta = grouped[grouped["balancer"] == method][f"{metric}_delta"].mean()
        plot_data.append({"Method": method, "Metric": metric, "Delta": mean_delta})

plot_df = pd.DataFrame(plot_data)

# Рисуем столбики
sns.barplot(data=plot_df, x="Metric", y="Delta", hue="Method", ax=ax)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Изменение метрик качества относительно baseline (среднее по 15 наборам)")
ax.set_ylabel("Разница (Метод - Baseline)")
ax.set_xlabel("Метрика качества")

plt.tight_layout()
plt.savefig("../../experiments_output/summary/quality_tradeoff.png", dpi=300)
print("График сохранен: quality_tradeoff.png")
