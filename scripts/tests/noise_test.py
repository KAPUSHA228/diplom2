from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from xai.explainers import get_shap_values
from xai.stability import rank_correlation

# Генерируем данные (как synth_10_r1)
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=5, weights=[0.9, 0.1], random_state=42, shuffle=False, flip_y=0
)

# Один и тот же тестовый сплит для всех
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Модель 1
rf1 = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1)
rf1.fit(X_train, y_train)
shap1 = get_shap_values(rf1, X_test)

# Модель 2 (другой seed)
rf2 = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=43, n_jobs=1)
rf2.fit(X_train, y_train)
shap2 = get_shap_values(rf2, X_test)

# Считаем шум (модель сама с собой)
noise_corr = rank_correlation(shap1, shap2)
print(f"Шум (Спирмен RF vs RF на одном сплите, разные seed): {noise_corr:.4f}")
