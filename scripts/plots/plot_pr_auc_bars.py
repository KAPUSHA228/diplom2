import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загружаем данные
df = pd.read_csv("../../experiments_output/summary/all_metrics.csv")

# 2. Проверяем, как называются колонки (на случай, если 'balancer' стал индексом)
if "balancer" not in df.columns:
    df = df.reset_index()

# Выводим для проверки, какие колонки есть на самом деле
print("Доступные колонки:", df.columns.tolist())

# 3. Фильтруем по датасету с дисбалансом 10% (r1, r2, r3)
df_10 = df[df["dataset"].str.startswith("synth_10")]

# 4. Группируем и считаем средний PR-AUC
pivot_df = df_10.groupby(["model", "balancer"])["pr_auc"].mean().unstack()

# 5. Оставляем только те методы, которые реально есть в данных
cols = [
    c
    for c in ["baseline", "class_weight", "smote", "borderline_smote", "adasyn", "smote_tomek", "smote_enn"]
    if c in pivot_df.columns
]
pivot_df = pivot_df[cols]

# 6. Строим график
sns.set_theme(style="whitegrid")

pivot_df.plot(kind="bar", figsize=(14, 6), colormap="viridis", edgecolor="black")

plt.title("Результаты по качеству: PR-AUC на датасете с дисбалансом 10%", fontsize=14)
plt.xlabel("Модель машинного обучения", fontsize=12)
plt.ylabel("Площадь под кривой точность-полнота (PR-AUC)", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Метод балансировки", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

out_path = "../../experiments_output/summary/pr_auc_comparison.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"График сохранен: {out_path}")
