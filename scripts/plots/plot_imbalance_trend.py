from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

runs_dir = Path("../../experiments_output/runs")
summary_dir = Path("../../experiments_output/summary")

# --- 1. СБОР И АГРЕГАЦИЯ ДАННЫХ ---
# Структура: {model: {ratio: {balancer: [avg_corr_r1, avg_corr_r2, avg_corr_r3]}}}
trend_data = {}
truth_set = set(range(7))  # 5 инф + 2 избыточных
truth_data = {}

for model in ["LR", "RF", "XGB"]:
    trend_data[model] = {}
    truth_data[model] = {}
    for ratio in [1, 5, 10, 20, 30]:
        trend_data[model][ratio] = {"SMOTE": [], "Class Weight": []}
        truth_data[model][ratio] = {"SMOTE": [], "Class Weight": []}

        for r_idx in range(1, 4):
            ds_name = f"synth_{ratio:02d}_r{r_idx}"
            # Ищем папки запусков для этого датасета
            dataset_runs = [
                d for d in runs_dir.iterdir() if d.is_dir() and f"_{ds_name}_" in d.name and f"_{model}_" in d.name
            ]

            for bal_name, label in [("smote", "SMOTE"), ("class_weight", "Class Weight")]:
                # Находим конкретный запуск
                target_run = next((r for r in dataset_runs if f"_{bal_name}_" in r.name), None)
                base_run = next((r for r in dataset_runs if "_baseline_" in r.name), None)

                if not target_run or not base_run:
                    continue

                try:
                    shap_target = np.load(target_run / "shap_values.npy")  # (15, n_test, features)
                    shap_base = np.load(base_run / "shap_values.npy")

                    imp_target = np.abs(shap_target).mean(axis=1)  # (15, features)
                    imp_base = np.abs(shap_base).mean(axis=1)

                    # Считаем корреляцию Спирмена для каждого из 15 разбиений
                    corrs = [spearmanr(imp_base[i], imp_target[i])[0] for i in range(15)]
                    trend_data[model][ratio][label].append(np.mean(corrs))  # Усредняем 15 разбиений для 1 набора

                    # Считаем истину (Top-5 hits) для каждого из 15 разбиений
                    hits = [len(set(np.argsort(-imp_target[i])[:5]) & truth_set) / 5.0 for i in range(15)]
                    truth_data[model][ratio][label].append(np.mean(hits))

                except Exception as e:
                    print(f"Ошибка чтения {target_run.name}: {e}")

# --- 2. СТАТИСТИКА (Уилкоксон по 3 наборам + Холм) ---
print("=== СТАТИСТИКА (Уилкоксон по 3 наборам) ===")
stats_rows = []
for model in ["LR", "RF", "XGB"]:
    for ratio in [1, 5, 10, 20, 30]:
        s_corr = trend_data[model][ratio]["SMOTE"]
        c_corr = trend_data[model][ratio]["Class Weight"]
        # У нас теперь ровно 3 значения (по одному на каждый набор r1, r2, r3)
        if len(s_corr) == 3 and len(c_corr) == 3:
            stat, p = wilcoxon(s_corr, c_corr)
            stats_rows.append(
                {"model": model, "ratio": ratio, "p_value": p, "delta_corr": np.mean(c_corr) - np.mean(s_corr)}
            )

if stats_rows:
    df_stats = pd.DataFrame(stats_rows)
    df_stats["p_holm"] = multipletests(df_stats["p_value"], method="holm")[1]
    print(df_stats.to_string(index=False))
    df_stats.to_csv(summary_dir / "final_stats.csv", index=False)

# --- 3. ГРАФИКИ ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 2 графика: Спирмен и Истина

# График 1: Корреляция Спирмена
ax = axes[0]
ratios = [1, 5, 10, 20, 30]
for model, color in zip(["LR", "RF", "XGB"], ["blue", "green", "red"]):
    smote_means = [np.mean(trend_data[model][r]["SMOTE"]) for r in ratios]
    cw_means = [np.mean(trend_data[model][r]["Class Weight"]) for r in ratios]
    ax.plot(ratios, smote_means, marker="o", linestyle="--", color=color, alpha=0.6, label=f"SMOTE ({model})")
    ax.plot(ratios, cw_means, marker="s", linestyle="-", color=color, label=f"Class Weight ({model})")

ax.set_title("Стабильность ранжирования (Корреляция Спирмена)")
ax.set_xlabel("Доля миноритарного класса (%)")
ax.set_ylabel("Корреляция с baseline")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# График 2: Истина (Top-5 hits)
ax = axes[1]
for model, color in zip(["LR", "RF", "XGB"], ["blue", "green", "red"]):
    smote_means = [np.mean(truth_data[model][r]["SMOTE"]) for r in ratios]
    cw_means = [np.mean(truth_data[model][r]["Class Weight"]) for r in ratios]
    ax.plot(ratios, smote_means, marker="o", linestyle="--", color=color, alpha=0.6, label=f"SMOTE ({model})")
    ax.plot(ratios, cw_means, marker="s", linestyle="-", color=color, label=f"Class Weight ({model})")

ax.set_title("Восстановление истинных признаков (Доля в Топ-5)")
ax.set_xlabel("Доля миноритарного класса (%)")
ax.set_ylabel("Доля истинных признаков в Топ-5")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig(summary_dir / "final_pipeline_analysis.png", dpi=300)
plt.show()
