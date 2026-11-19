import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("../../experiments_output/summary/per_split_metrics.csv")
# Оставляем только синтетику
df = df[df["dataset"].str.match(r"^synth_\d+_r\d$")]

stats_rows = []
for model in ["LR", "RF", "XGB"]:
    sub = df[(df["model"] == model) & (df["balancer"].isin(["smote", "class_weight"]))]

    # Усредняем по 15 разбиениям для каждого набора
    grouped = sub.groupby(["dataset", "balancer"])["sign_agreement"].mean().unstack()

    if "smote" in grouped.columns and "class_weight" in grouped.columns:
        delta = grouped["class_weight"] - grouped["smote"]
        stat, p = wilcoxon(grouped["class_weight"], grouped["smote"])

        stats_rows.append({"model": model, "mean_delta_signs": delta.mean(), "p_value": p})

df_stats = pd.DataFrame(stats_rows)
df_stats["p_holm"] = multipletests(df_stats["p_value"], method="holm")[1]
print("\n=== СОВПАДЕНИЕ ЗНАКОВ SHAP (Class Weight - SMOTE) ===")
print(df_stats.to_string(index=False))
