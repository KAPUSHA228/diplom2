import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

print("Загрузка датасета UCI...")
dataset = fetch_ucirepo(id=697)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

# 1. Оставляем только Dropout и Graduate (Enrolled удаляем)
df = df[df["Target"] != "Enrolled"].copy()
df["risk_flag"] = (df["Target"] == "Dropout").astype(int)
df = df.drop(columns=["Target"])

# 2. УБИРАЕМ УТЕЧКУ (удаляем признаки 2-го семестра)
cols_to_drop = [c for c in df.columns if "2nd sem" in c or "2nd_sem" in c]
df = df.drop(columns=cols_to_drop)

# 3. Кодируем категориальные признаки
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 4. Искусственно создаем дисбаланс 90/10
# Оставляем всех Graduate (класс 0) и случайно сэмплируем Dropout (класс 1) до 10%
n_grad = len(df[df["risk_flag"] == 0])
target_n_dropout = int(n_grad * 0.1 / 0.9)
df_dropout = df[df["risk_flag"] == 1].sample(n=target_n_dropout, random_state=42)
df_grad = df[df["risk_flag"] == 0]
df_final = pd.concat([df_grad, df_dropout]).sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Сохраняем
out_path = Path("../../data") / "uci_student_dropout_prepared.csv"
df_final.to_csv(out_path, index=False)

print(f"\nГотово! Файл сохранён: {out_path}")
print("Форма:", df_final.shape)
print("Дисбаланс (риск / успех):")
print(df_final["risk_flag"].value_counts(normalize=True).round(3))
