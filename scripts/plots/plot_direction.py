import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Загружаем данные (Вот этого не хватало!)
summary_dir = Path("../../experiments_output/summary")
df = pd.read_csv(summary_dir / "per_split_metrics.csv")

# Оставляем только синтетику с генерациями r1, r2, r3
df = df[df["dataset"].str.match(r"^synth_\d+_r\d$")]

# 2. Настраиваем график
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
ratios = [1, 5, 10, 20, 30]

# 3. Считаем и рисуем
for model, color in zip(["LR", "RF", "XGB"], ["blue", "green", "red"]):
    smote_means, cw_means = [], []
    for r in ratios:
        # Берем данные для конкретной доли (например, synth_01) и модели
        sub_smote = df[
            (df["dataset"].str.startswith(f"synth_{r:02d}")) & (df["model"] == model) & (df["balancer"] == "smote")
        ]
        sub_cw = df[
            (df["dataset"].str.startswith(f"synth_{r:02d}"))
            & (df["model"] == model)
            & (df["balancer"] == "class_weight")
        ]

        smote_means.append(sub_smote["sign_agreement"].mean())
        cw_means.append(sub_cw["sign_agreement"].mean())

    # Рисуем пунктир для SMOTE и сплошную для Class Weight
    ax.plot(ratios, smote_means, marker="o", linestyle="--", color=color, alpha=0.6, label=f"SMOTE ({model})")
    ax.plot(ratios, cw_means, marker="s", linestyle="-", color=color, label=f"Class Weight ({model})")

# 4. Оформление
ax.set_title("Совпадение знаков SHAP с базовой моделью (Direction Fidelity)")
ax.set_xlabel("Доля миноритарного класса (%)")
ax.set_ylabel("Доля совпавших знаков")
ax.set_ylim(0.4, 1.05)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# 5. Сохранение
out_path = summary_dir / "direction_fidelity.png"
plt.savefig(out_path, dpi=300)
print(f"График сохранен: {out_path}")
